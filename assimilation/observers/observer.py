import numpy as np
from numpy.typing import ArrayLike
import torch

"""
Base class for Observer object

Input is  generated data, returns ObsVector with values, times, coords, etc
"""


class Observer():
    """
    Base class for sampling observations from generated data.

    Args:
        obs_fun (callable): Function to apply to the state vector to produce observations.
        obs_prob (float or tuple of float, optional): Fraction of spatial locations to randomly select for observing. Must be between 0 and 1. Default is 1.
        obs_mask (array-like, optional): Manually specified indices for observing. If 1D, assumed to be for flattened spatial dimensions. If multi-dimensional, must match the shape of the data generator's output. If not specified, indices are randomly generated according to random_location_density. Default is None.
        stationary_obs (bool, optional): If True, uses the same observation indices at each time step. If False, generates new indices at each time step. Default is True.
        random_seed (int, optional): Random seed for reproducibility when sampling observation locations. Default is 42.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.

    Attributes:
        obs_prob (float or tuple of float): Fraction of locations to observe.
        obs_mask (array-like or None): Indices specifying observed locations.
        stationary_obs (bool): Whether observation indices are fixed over time.
        random_seed (int): Seed for random number generation.
    """

    def __init__(self,
                 obs_fun: callable,
                 obs_prob: float = 1.,
                 obs_mask: ArrayLike = None,
                 stationary_obs: bool = True,
                 random_seed: int = 42,
                 nx: int = 64,
                 ny: int = 64,
                 device: torch.device = torch.device('cpu'),
                 num_channels: int = 2,
                 unit_factor: float = 1.0,
                 ):
        # Channels per grid point and raw -> physical factor (DatasetMetadata).
        self.num_channels = num_channels
        self.unit_factor = unit_factor
        self.obs_fn = obs_fun
        self.obs_prob = obs_prob
        self.obs_mask = obs_mask  # TODO: Needs to be converted to indxob
        if obs_mask is not None:
            raise NotImplementedError(
                "Manually specified obs_mask is not implemented yet.")
        self.stationary_obs = stationary_obs
        self.random_seed = random_seed
        self.nx = nx
        self.ny = ny
        self.nobs = int(nx * ny * self.obs_prob)
        self.rsobs = np.random.RandomState(
            self.random_seed)  # fixed seed for observations
        self.indxob = np.sort(self.rsobs.choice(
            nx*ny, self.nobs, replace=False))

        self.device = device

    def observe(self, state_vector):
        """
        Apply the observation operator to a state vector.

        Args:
            state_vector: The state vector to observe.

        Returns:
            obs_vector: The observed values after applying the observation operator.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class GridObserver(Observer):
    """
    Observer for SQG (Surface Quasi-Geostrophic) data assimilation experiments.

   Args:
        obs_fun (callable): Function to apply to the state vector to produce observations.
        obs_prob (float or tuple of float, optional): Fraction of spatial locations to randomly select for observing. Must be between 0 and 1. Default is 1.
        obs_mask (array-like, optional): Manually specified indices for observing. If 1D, assumed to be for flattened spatial dimensions. If multi-dimensional, must match the shape of the data generator's output. If not specified, indices are randomly generated according to random_location_density. Default is None.
        stationary_obs (bool, optional): If True, uses the same observation indices at each time step. If False, generates new indices at each time step. Default is True.
        random_seed (int, optional): Random seed for reproducibility when sampling observation locations. Default is 42.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
    """

    def __init__(self,
                 obs_fun: callable,
                 obs_prob: float = None,
                 obs_sigma: float = None,
                 obs_mask: ArrayLike = None,
                 stationary_obs: bool = True,
                 random_seed: int = 42,
                 nx: int = 64,
                 ny: int = 64,
                 device: torch.device = torch.device('cpu'),
                 radar: bool = False,
                 radar_width: int = 4,
                 radar_dx: int = 2,
                 radar_x0: int = None,
                 num_channels: int = 2,
                 unit_factor: float = 1.0,
                 ):
        self.obs_sigma = obs_sigma
        self.radar = radar
        self.radar_width = radar_width
        self.radar_dx = radar_dx
        self.radar_x0 = radar_x0 if radar_x0 is not None else 0

        super().__init__(obs_fun, obs_prob, obs_mask,
                         stationary_obs, random_seed, nx, ny, device,
                         num_channels, unit_factor)

    def observe(self, x_state, t=None):
        """
        Apply the observation operator to a state vector.

        Args:
            x_state: The state vector to observe.

        Returns:
            obs_vector: The observed values after applying the observation operator.
        """

        if not self.stationary_obs or t == 0:

            if self.radar:
                # --- build grid ---
                I, J = np.meshgrid(
                    np.arange(self.ny),
                    np.arange(self.nx),
                    indexing="ij"
                )

                # --- radar center moves with time ---
                x_left = (self.radar_x0 + t * self.radar_dx) % self.nx

                # --- periodic distance in x ---
                dx = (J - x_left) % self.nx

                # --- vertical radar mask ---
                radar_mask = dx < self.radar_width

                self.indxob = np.sort((I * self.nx + J)[radar_mask].ravel())
                self.nobs = len(self.indxob)

            elif self.nobs == self.nx * self.ny:
                self.indxob = np.arange(self.nx * self.ny)

            else:
                self.indxob = np.sort(self.rsobs.choice(
                    self.nx * self.ny, self.nobs, replace=False
                ))

        n_ch = self.num_channels
        pvob = np.empty((n_ch, self.nobs), float)

        for k in range(n_ch):
            pvob[k] = self.unit_factor * \
                x_state[k, :, :].ravel()[self.indxob]
            pvob[k] = self.obs_fn(torch.tensor(pvob[k])).numpy(
            ) + self.rsobs.normal(scale=self.obs_sigma, size=int(self.nobs))

        cell = self.nx * self.ny
        indxob_ensf = np.concatenate(
            [self.indxob + k * cell for k in range(n_ch)], axis=None)

        obs_input = pvob.reshape(n_ch*self.nobs)
        sparse_idx = indxob_ensf

        sparse_index = torch.zeros(
            n_ch*cell, dtype=torch.bool, device=self.device)
        sparse_index[sparse_idx] = True

        obs_mask = torch.zeros(
            1, n_ch*cell, dtype=torch.bool, device=self.device)  # bool mask
        obs_mask[:, sparse_index] = True  # set observed indices to True
        obs_mask = obs_mask.view(1, n_ch, self.ny, self.nx)

        obs = torch.tensor(obs_input, device=self.device,
                           dtype=torch.float32).unsqueeze(0)
        obs_sigma = self.obs_sigma

        return obs, obs_mask, obs_sigma


class AvgObserver(Observer):
    """
    Observer for SQG (Surface Quasi-Geostrophic) data assimilation experiments.

   Args:
        obs_fun (callable): Function to apply to the state vector to produce observations.
        obs_prob (float or tuple of float, optional): Fraction of spatial locations to randomly select for observing. Must be between 0 and 1. Default is 1.
        obs_mask (array-like, optional): Manually specified indices for observing. If 1D, assumed to be for flattened spatial dimensions. If multi-dimensional, must match the shape of the data generator's output. If not specified, indices are randomly generated according to random_location_density. Default is None.
        stationary_obs (bool, optional): If True, uses the same observation indices at each time step. If False, generates new indices at each time step. Default is True.
        random_seed (int, optional): Random seed for reproducibility when sampling observation locations. Default is 42.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
    """

    def __init__(self,
                 obs_fun: callable,
                 obs_prob: float = None,
                 obs_sigma: float = None,
                 obs_mask: ArrayLike = None,
                 stationary_obs: bool = True,
                 random_seed: int = 42,
                 nx: int = 64,
                 ny: int = 64,
                 device: torch.device = torch.device('cpu'),
                 lres: int = 64,
                 num_channels: int = 2,
                 unit_factor: float = 1.0,
                 ):
        super().__init__(obs_fun, obs_prob, obs_mask,
                         stationary_obs, random_seed, nx, ny, device,
                         num_channels, unit_factor)

        self.nobs = int(lres * lres * self.obs_prob)
        self.rsobs = np.random.RandomState(
            self.random_seed)  # fixed seed for observations
        self.lres_indxob = np.sort(self.rsobs.choice(
            lres * lres, self.nobs, replace=False))

        assert self.stationary_obs == True, "AvgObserver only supports non-stationary observations for now."

        self.obs_sigma = obs_sigma
        self.lres = lres
        self.r = self.nx // self.lres
        self.nx = nx

        self.idx_centers = self.get_centers()
        self.indxob = self.idx_centers[self.lres_indxob]
        self.ii_flat, self.jj_flat = self.get_indices()

        self.obs_fn = self.avg_obs_fn

    def get_centers(self):
        # Coarse-grid indices (centroids)
        idx_centers = []
        for i in range(self.lres):
            for j in range(self.lres):
                ci = i * self.r + self.r//2
                cj = j * self.r + self.r//2
                idx_centers.append(ci*self.nx + cj)
        idx_centers = torch.tensor(idx_centers)
        return idx_centers

    def get_indices(self):
        # Get the indices for the observed boxes

        # Convert idx_centers to 2D coordinates
        i_cent = self.indxob // self.nx  # shape (nobs,)
        j_cent = self.indxob % self.nx   # shape (nobs,)

        # Compute offsets within box
        offsets = torch.arange(-self.r//2, self.r//2)  # shape (r,)

        # Broadcast to create all indices for each box
        ii = (i_cent[:, None] + offsets[None, :]) % self.nx  # shape (nobs, r)
        jj = (j_cent[:, None] + offsets[None, :]) % self.nx  # shape (nobs, r)

        # Create a grid of all combinations within each box
        # ii_grid: (nobs, r, r), jj_grid: (nobs, r, r)
        ii_grid = ii[:, :, None].expand(self.nobs, self.r, self.r)
        jj_grid = jj[:, None, :].expand(self.nobs, self.r, self.r)

        # Flatten last two dims to index into x
        ii_flat = ii_grid.reshape(self.nobs, -1)  # (nobs, r*r)
        jj_flat = jj_grid.reshape(self.nobs, -1)  # (nobs, r*r)
        return ii_flat, jj_flat

    def avg_obs_fn(self, x):
        """
        Vectorized box-averaged observations for arbitrary box centroids.

        Args:
            x: torch.Tensor, shape (B, C, nx, nx)
            idx_centers: torch.Tensor or array of shape (nobs,), flattened indices of box centers
            r: int, box size

        Returns:
            obs: torch.Tensor, shape (B, C, nobs), averaged values for each box
        """

        # Gather all values and compute mean
        # x: (C, nx, nx) -> (C, nobs, r*r)
        if x.dim() == 3:
            vals = x[:, self.ii_flat, self.jj_flat]  # advanced indexing
        elif x.dim() == 4:
            vals = x[:, :, self.ii_flat, self.jj_flat]  # advanced indexing

        obs = vals.mean(dim=-1)           # (C, nobs)
        return obs

    def observe(self, x_state, t=None):
        """
        Apply the observation operator to a state vector.

        `t` is ignored; it keeps the signature shared with GridObserver.

        Args:
            x_state: The state vector to observe.

        Returns:
            obs_vector: The observed values after applying the observation operator.
        """
        n_ch = self.num_channels

        pvob = self.obs_fn(torch.tensor(
            self.unit_factor * x_state)).numpy()
        pvob += self.rsobs.normal(scale=self.obs_sigma,
                                  size=int(self.nobs*n_ch)).reshape(n_ch, self.nobs)

        cell = self.nx * self.ny
        indxob_ensf = np.concatenate(
            [self.indxob + k * cell for k in range(n_ch)], axis=None)

        obs_input = pvob
        sparse_idx = indxob_ensf

        sparse_index = torch.zeros(
            n_ch*cell, dtype=torch.bool)
        sparse_index[sparse_idx] = True

        obs_mask = torch.zeros(
            1, n_ch*cell, dtype=torch.bool)
        obs_mask[:, sparse_index] = True  # set observed indices to True
        obs_mask = obs_mask.view(1, n_ch, self.ny, self.nx)

        obs = torch.tensor(obs_input,
                           dtype=torch.float32)
        obs_sigma = self.obs_sigma

        return obs, obs_mask, obs_sigma


# Backward-compatible alias.
SQGObserver = GridObserver
