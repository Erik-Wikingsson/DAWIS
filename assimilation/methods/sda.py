import numpy as np
import torch
from tqdm import tqdm

from assimilation.methods.dawis import DAWIS


class SDA(DAWIS):
    """
    Score-based Data Assimilation.

    This mirrors the DAWIS setup, but assembles a global trajectory drift by
    evaluating overlapping windows from the same trajectory state.
    """

    def __init__(self, args, metadata=None):
        # `metadata` lets DAWIS resolve the window model; None requires --model_path.
        super().__init__(args, metadata=metadata)
        self.method_name = "SDA"
        self.sampler_eps = getattr(args, "sampler_eps", 0.0)
        self.corrections = getattr(args, "corrections", 0)
        self.tau = getattr(args, "tau", 0.5)

        if args.guide_method == 'MMPS' and args.obs_fn == 'arctan':
            def GUIDANCE(t): return args.guidance_strength * (1-t)
        else:
            def GUIDANCE(t):
                return args.guidance_strength * torch.ones_like(t) 

        self.set_guidance_strength(GUIDANCE)

        self.window_batch_size = getattr(args, "window_batch_size", 5)

        # Ensemble members per network call (same meaning as for DAWIS);
        # applied inside get_drift_and_score. None = whole ensemble at once.
        self.batch_size = getattr(args, "batch_size", None)

        print(
            f"SDA initialized | device={self.device} | fm_loss={getattr(args, 'fm_loss', 'eta01_label')} | "
            f"window={self.init_states + 1} | guide={getattr(args, 'guide_method', None)} | "
            f"window_batch_size={self.window_batch_size} | batch_size={self.batch_size}")

    @staticmethod
    def _window_bounds(t_global, total_length, window_size):
        half_window = window_size // 2
        if t_global <= half_window:
            left = 0
            right = window_size
        elif t_global >= total_length - half_window - 1:
            left = total_length - window_size
            right = total_length
        else:
            left = t_global - half_window
            right = t_global + half_window + 1
        return left, right

    def _make_trajectory_obs_fn(self, obs_fn, scale, obs_mask_window=None, avg=False,
                                obs_fn_window=None):
        """
        Build an observation operator for a full window.

        The returned function maps `(B, T, D, X, Y)` to one concatenated
        observation vector per time step in the window.

        `obs_fn_window` optionally gives one operator per window slot, so a slot
        can be observed differently from the rest (e.g. the identity, full-field
        initial-condition observation used by `--guide_first`). When it is None
        every slot uses `obs_fn`.
        """

        def _flatten_obs(obs_tensor):
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            return obs_tensor.reshape(obs_tensor.shape[0], -1)

        def _obs_fn(z_norm):
            x_sf = z_norm * scale
            step_obs = []
            for t_idx in range(x_sf.shape[1]):
                x_step = x_sf[:, t_idx]
                step_fn = obs_fn if obs_fn_window is None else obs_fn_window[t_idx]
                if avg:
                    obs_step = step_fn(x_step)
                elif obs_mask_window is not None:
                    obs_step = step_fn(x_step[:, obs_mask_window[t_idx]])
                else:
                    obs_step = step_fn(x_step)
                step_obs.append(_flatten_obs(obs_step))
            return torch.cat(step_obs, dim=1)

        return _obs_fn

    def smooth(self, smooth_args, **kwargs):
        """
        Run SDA over a full trajectory.

        Args:
            smooth_args: namespace-like object with trajectory, observation, and
                method settings.
        """
        obs_history = smooth_args.obs_history
        obs_mask_history = smooth_args.obs_mask_history
        obs_fn = smooth_args.obs_fn
        n_times = smooth_args.n_times
        window_radius = smooth_args.window_radius
        steps = smooth_args.steps
        avg = smooth_args.avg
        trajectory_init = smooth_args.trajectory_init
        return_traj = getattr(smooth_args, "return_traj", False)
        # Optional per-step obs operator / noise std (used by --guide_first).
        obs_fn_history = getattr(smooth_args, "obs_fn_history", None)
        obs_sigma_history = getattr(smooth_args, "obs_sigma_history", None)

        device = self.device
        scale = self.scale
        n_ens = self.members
        window_size = 2 * window_radius + 1
        expected_window = self.init_states + 1
        eps = self.sampler_eps
        if expected_window != window_size:
            raise ValueError(
                f"SDA window size {window_size} does not match model window size {expected_window}. "
                "Set `init_states = 2 * sda_w` before constructing the SDA method.")

        if steps is None:
            steps = self.sampler_steps

        if trajectory_init is None:
            z = torch.randn((n_ens, n_times, *self.model.md.state_shape), device=device)
        else:
            z = torch.as_tensor(trajectory_init, device=device, dtype=torch.float32)
            if z.dim() == 4:
                z = z.unsqueeze(0).repeat(n_ens, 1, 1, 1, 1)
            elif z.dim() != 5:
                raise ValueError(
                    f"trajectory_init must have shape (T, D, X, Y) or (n_ens, T, D, X, Y), got {tuple(z.shape)}")

        ts = torch.linspace(0.0, 1.0, steps + 1, device=device)[:-1]
        dt = 1.0 / steps

        windows = []
        for t_global in range(n_times):
            left, right = self._window_bounds(t_global, n_times, window_size)
            windows.append((t_global, left, right))
        print('windows',windows)

        # Group windows that share an observation operator.
        groups = {}

        for t_global, left, right in windows:
            if avg:
                key = (right - left, "avg")
                mask_arr = None
            else:
                masks = [obs_mask_history[i].squeeze(0).cpu().numpy()
                        for i in range(left, right)]
                mask_arr = np.stack(masks, axis=0)
                key = mask_arr.tobytes()

            # Per-step operators/sigmas are part of the key.
            obs_fn_window = (None if obs_fn_history is None
                             else [obs_fn_history[i] for i in range(left, right)])
            sigma_window = (None if obs_sigma_history is None
                            else [float(obs_sigma_history[i]) for i in range(left, right)])
            key = (key,
                   None if obs_fn_window is None else tuple(id(f) for f in obs_fn_window),
                   None if sigma_window is None else tuple(sigma_window))

            groups.setdefault(key, {
                "mask": mask_arr,
                "obs_fn_window": obs_fn_window,
                "sigma_window": sigma_window,
                "entries": []
            })

            # Blocks with a leading ensemble dim are per-member (--guide_first
            # x0 slot); such windows get an (n_ens, n_obs_total) obs matrix.
            blocks = []
            for i in range(left, right):
                block = obs_history[i].to(device)
                if block.dim() >= 2 and block.shape[0] == n_ens and n_ens > 1:
                    blocks.append(block.reshape(n_ens, -1))
                else:
                    blocks.append(block.flatten())

            if any(b.dim() == 2 for b in blocks):
                obs_vec = torch.cat(
                    [b if b.dim() == 2 else b.unsqueeze(0).expand(n_ens, -1)
                     for b in blocks],
                    dim=1
                )
            else:
                obs_vec = torch.cat(blocks, dim=0)

            groups[key]["entries"].append(
                (t_global, left, right, obs_vec)
            )

        for t_scalar in tqdm(ts):
            # Corrector
            for _ in range(self.corrections):
                drift, score = self.get_drift_and_score(z, groups, obs_fn, scale, device, t_scalar, avg=avg)
                delta_t = self.tau / score.square().mean(dim=(2, 3, 4), keepdim=True)
                dz = score * delta_t
                dW = torch.randn_like(z) * torch.sqrt(2 * delta_t)

                z = z + dz + dW

            # Predictor
            drift, score = self.get_drift_and_score(z, groups, obs_fn, scale, device, t_scalar, avg=avg)
            eps_t = self.model.schedule.alpha(t_scalar) * eps
            dW = torch.sqrt(2 * dt * eps_t) * torch.randn_like(z)
            z = z + dt * drift + dW

        if return_traj:
            return z
        return z

    def get_drift_and_score(
        self,
        z,
        groups,
        obs_fn,
        scale,
        device,
        t_scalar,
        avg=False,
    ):

        drift = torch.zeros_like(z)
        score = torch.zeros_like(z)

        chunk_size = self.window_batch_size
        n_ens = z.shape[0]

        # Windows and members are chunked independently: each forward pass
        # sees up to chunk_size * ens_chunk windows.
        ens_chunk = n_ens if self.batch_size is None else self.batch_size
        ens_chunk = max(1, min(int(ens_chunk), n_ens))

        for group in groups.values():
            mask0 = group["mask"]

            obs_mask_window_tensor = (
                None if avg else
                torch.tensor(mask0, device=device, dtype=torch.bool)
            )

            self.model.observation_fn = self._make_trajectory_obs_fn(
                obs_fn,
                scale,
                obs_mask_window_tensor,
                avg=avg,
                obs_fn_window=group.get("obs_fn_window"),
            )

            self.model.obs_sigma = self._group_obs_sigma(
                group, obs_mask_window_tensor, device)

            entries = group["entries"]

            # Chunking does not change the result; batch_map records where each row goes.
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start:start + chunk_size]

                for ens_start in range(0, n_ens, ens_chunk):
                    ens_end = min(ens_start + ens_chunk, n_ens)

                    z_batch = []
                    obs_batch = []
                    batch_map = []

                    for t_global, left, right, obs_vec in chunk:
                        for ens_idx in range(ens_start, ens_end):

                            z_batch.append(
                                z[ens_idx, left:right].unsqueeze(0)
                            )

                            obs_batch.append(
                                # (n_ens, n_obs) -> this member's row; (n_obs,) is shared
                                obs_vec[ens_idx:ens_idx + 1] if obs_vec.dim() == 2
                                else obs_vec.unsqueeze(0)
                            )

                            batch_map.append(
                                (ens_idx, t_global, t_global - left)
                            )

                    z_batch = torch.cat(z_batch, dim=0)
                    obs_batch = torch.cat(obs_batch, dim=0)

                    t_param = torch.full(
                        (z_batch.shape[0], z_batch.shape[1]),
                        t_scalar,
                        device=device,
                    )

                    with torch.no_grad():

                        drift_local, score_local = self.model.get_drift_and_score(
                            z_batch,
                            t_param,
                            eps=self.sampler_eps,
                            schedule=self.model.schedule,
                            obs=obs_batch,
                        )

                    for batch_idx, (ens_idx, t_global, center_idx) in enumerate(batch_map):

                        drift[ens_idx, t_global] = drift_local[
                            batch_idx,
                            center_idx,
                        ]

                        score[ens_idx, t_global] = score_local[
                            batch_idx,
                            center_idx,
                        ]

                    del (
                        z_batch,
                        obs_batch,
                        t_param,
                        drift_local,
                        score_local,
                        batch_map,
                    )

        return drift, score

    def _group_obs_sigma(self, group, obs_mask_window, device):
        """Observation-noise std aligned with a group's concatenated obs vector.

        Returns the scalar `self.obs_sigma`, or, if per-step sigmas are given, a
        (1, n_obs_total) tensor repeating each step's sigma over its observed
        elements.
        """
        sigma_window = group.get("sigma_window")
        if sigma_window is None:
            return self.obs_sigma

        cached = group.get("sigma_vec")
        if cached is not None:
            return cached

        if obs_mask_window is None:
            raise ValueError(
                "Per-step obs_sigma is not supported with averaged observations "
                "(there is no per-step mask to align the sigma vector with).")

        blocks = [
            torch.full((int(obs_mask_window[t].sum()),), sigma_window[t],
                       device=device, dtype=torch.float32)
            for t in range(len(sigma_window))
        ]
        sigma_vec = torch.cat(blocks, dim=0).unsqueeze(0)
        group["sigma_vec"] = sigma_vec
        return sigma_vec

    def assimilate(self, *args, **kwargs):
        return self.smooth(*args, **kwargs)