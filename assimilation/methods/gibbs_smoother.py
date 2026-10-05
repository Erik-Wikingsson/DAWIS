import torch
from tqdm import tqdm

from assimilation.methods.dawis import DAWIS, _setdefault

class BlockGibbsSmoother(DAWIS):
    """Block Gibbs smoother built on top of DAWIS."""

    def __init__(self, args, metadata=None):
        # `metadata` is passed through to DAWIS to resolve the FMW checkpoint.
        _setdefault(args, "gibbs_sweeps", 1)
        _setdefault(args, "block_mode", "sliding")
        _setdefault(args, "block_direction", "forward")
        _setdefault(args, "block_stride", None)
        _setdefault(args, "random_block_count", None)
        _setdefault(args, "random_block_replace", False)
        _setdefault(args, "gibbs_endpoint_tmin", 1.0)
        _setdefault(args, "gibbs_midpoint_tmin", 0.0)
        _setdefault(args, "gibbs_num_endpoints", 1)

        super().__init__(args, metadata=metadata)

        if args.guide_method == 'MMPS' and args.obs_fn == 'arctan':
            def GUIDANCE(t): return args.guidance_strength * (1-t)
            print("Using MMPS guidance with arctan obs_fn")
        else:
            def GUIDANCE(t):
                return args.guidance_strength * torch.ones_like(t) 

        self.set_guidance_strength(GUIDANCE)

        self.gibbs_sweeps = getattr(args, "gibbs_sweeps", 1)
        self.block_mode = getattr(args, "block_mode", "sliding")
        self.block_direction = getattr(args, "block_direction", "forward")
        self.block_stride = getattr(args, "block_stride", None)
        self.random_block_count = getattr(args, "random_block_count", None)
        self.random_block_replace = getattr(args, "random_block_replace", False)
        self.gibbs_endpoint_tmin = getattr(args, "gibbs_endpoint_tmin", 1.0)
        self.gibbs_midpoint_tmin = getattr(args, "gibbs_midpoint_tmin", 0.0)
        # Conditioning steps held at the endpoint tmin on each side of a block
        # (1 = Markovian; a model with k-step memory needs k).
        self.gibbs_num_endpoints = int(getattr(args, "gibbs_num_endpoints", 1))
        if self.gibbs_num_endpoints < 1:
            raise ValueError(
                "gibbs_num_endpoints must be at least 1; got "
                f"{self.gibbs_num_endpoints}.")

        # --batch_size: ensemble members per block update (None = all at once).
        self.ens_batch_size = getattr(args, "batch_size", None)

        self.init_states = self.args.init_states
        if self.init_states <= 1:
            raise ValueError(
                "BlockGibbsSmoother expects init_states>1 so that the "
                f"interior is non-empty; got init_states={self.init_states}.")
        if self.init_states % 2 != 0:
            raise ValueError(
                "BlockGibbsSmoother expects init_states to be even so that the "
                f"block length is odd; got init_states={self.init_states}.")

        self.block_length = self.init_states + 1
        self.block_radius = self.init_states // 2
        if 2 * self.gibbs_num_endpoints >= self.block_length:
            raise ValueError(
                f"gibbs_num_endpoints={self.gibbs_num_endpoints} leaves no free "
                f"interior in a block of length {self.block_length}; it needs "
                f"2*gibbs_num_endpoints < init_states + 1.")
        self.method_name = "GIBBS"
        print(
            f"GIBBS initialized | device={self.device} | fm_loss={getattr(self.args, 'fm_loss', None)} | "
            f"block={self.block_length} | endpoints={self.gibbs_num_endpoints} | "
            f"mode={self.block_mode} | direction={self.block_direction}")
    @staticmethod
    def _flatten_obs(obs_tensor):
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor.reshape(obs_tensor.shape[0], -1)

    def _make_block_obs_fn(self, obs_fn, scale, obs_mask_window=None, obs_step_indices=None, avg=False):
        if obs_step_indices is None:
            obs_step_indices = range(obs_mask_window.shape[0] if obs_mask_window is not None else 0)

        def _obs_fn(z_norm):
            x_sf = z_norm * scale
            step_obs = []
            for t_idx in obs_step_indices:
                x_step = x_sf[:, t_idx]
                if avg:
                    obs_step = obs_fn(x_step)
                elif obs_mask_window is not None:
                    obs_step = obs_fn(x_step[:, obs_mask_window[t_idx]])
                else:
                    obs_step = obs_fn(x_step)
                step_obs.append(self._flatten_obs(obs_step))
            if not step_obs:
                return torch.zeros(x_sf.shape[0], 0, device=x_sf.device, dtype=x_sf.dtype)
            return torch.cat(step_obs, dim=1)

        return _obs_fn

    def _member_chunks(self, n_ens):
        """(start, stop) member slices of at most `--batch_size` each.

        Members are independent within a block update, so chunking bounds
        memory without changing the computation (only the RNG stream differs).
        """
        size = self.ens_batch_size
        if size is None or size <= 0 or size >= n_ens:
            return [(0, n_ens)]
        return [(i, min(i + size, n_ens)) for i in range(0, n_ens, size)]

    @staticmethod
    def _build_step_tmin(block_length, mid_tmin=0.0, left_tmin=1.0, right_tmin=1.0,
                         n_endpoints=1):
        """Per-step tmin for one block.

        The first and last `n_endpoints` steps are conditioning states held at
        the endpoint tmin; the interior is resampled at `mid_tmin`.
        """
        if block_length <= 0:
            raise ValueError(f"block_length must be positive, got {block_length}")
        n_endpoints = int(n_endpoints)
        if n_endpoints < 1:
            raise ValueError(f"n_endpoints must be at least 1, got {n_endpoints}")
        if block_length == 1:
            return [float(max(left_tmin, right_tmin))]
        # Where the two endpoint groups overlap, take the larger tmin.
        left_n = min(n_endpoints, block_length)
        right_n = min(n_endpoints, block_length)
        step_tmin = [float(mid_tmin)] * block_length
        for i in range(left_n):
            step_tmin[i] = float(left_tmin)
        for i in range(right_n):
            idx = block_length - 1 - i
            if idx < left_n:
                step_tmin[idx] = float(max(step_tmin[idx], right_tmin))
            else:
                step_tmin[idx] = float(right_tmin)
        return step_tmin

    @staticmethod
    def _normalize_custom_block(block, block_radius, block_length):
        if isinstance(block, int):
            return block
        if len(block) != 2:
            raise ValueError(
                f"Custom block spec must be an int center or a (left, right) pair; got {block}")
        left, right = int(block[0]), int(block[1])
        if right - left != block_length:
            raise ValueError(
                f"Custom block ({left}, {right}) must span exactly block_length={block_length} time steps.")
        return left + block_radius

    def _resolve_block_centers(self, n_times, window_radius, blocks=None, block_mode=None, block_stride=None):
        valid_centers = list(range(window_radius, n_times - window_radius))
        print('valid_centers', valid_centers)
        if not valid_centers:
            return []

        if blocks is not None:
            centers = [
                self._normalize_custom_block(block, window_radius, self.block_length)
                for block in blocks
            ]
            centers = [center for center in centers if center in valid_centers]
            return centers

        mode = block_mode or self.block_mode
        if mode == "sliding":
            stride = 1 if block_stride is None or block_stride == 0 else block_stride
            centers = valid_centers[::stride]
        elif mode == "non_overlapping":
            stride = self.block_length if block_stride is None or block_stride == 0 else block_stride
            centers = valid_centers[::stride]
        elif mode == "random":
            count = self.random_block_count or len(valid_centers)
            if self.random_block_replace:
                draw = torch.randint(0, len(valid_centers), (count,), device=self.device)
                centers = torch.tensor(valid_centers, device=self.device)[draw].tolist()
            else:
                perm = torch.randperm(len(valid_centers), device=self.device)
                centers = torch.tensor(valid_centers, device=self.device)[perm[: min(count, len(valid_centers))]].tolist()
        elif mode == "custom":
            raise ValueError("block_mode='custom' requires blocks to be passed to smooth().")
        else:
            raise ValueError(
                f"Unknown block_mode '{mode}'. Expected sliding, non_overlapping, random, or custom.")

        centers = [center for center in centers if center in valid_centers]

        # A stride wider than the resampled interior leaves steps never updated.
        interior = self.block_length - 2 * self.gibbs_num_endpoints
        if len(centers) > 1:
            gaps = [b - a for a, b in zip(centers, centers[1:])]
            if gaps and max(gaps) > interior:
                print(f"Warning: block spacing {max(gaps)} exceeds the resampled "
                      f"interior ({interior} steps) with "
                      f"gibbs_num_endpoints={self.gibbs_num_endpoints}; some time "
                      "steps are never updated.")
        return centers

    def _ordered_centers(self, centers, block_direction=None, sweep_count=0):
        direction = block_direction or self.block_direction
        if direction == "forward":
            return list(centers)
        if direction == "backward":
            return list(reversed(centers))
        if direction == "both":
            return list(reversed(centers)) + list(centers)
        if direction == "alternate":
            if sweep_count % 2 == 1:
                return list(centers)
            else:
                return list(reversed(centers))
        raise ValueError(
            f"Unknown block_direction '{direction}'. Expected forward, backward, or both.")

    def gibbs_step(
        self,
        trajectory,
        block_start,
        block_stop,
        obs_history,
        obs_mask_history,
        obs_fn,
        steps=None,
        avg=False,
    ):
        device = self.device
        scale = self.scale
        n_ens = trajectory.shape[0]
        steps = self.sampler_steps if steps is None else steps
        n_times = trajectory.shape[1]
        block = trajectory[:, block_start:block_stop].clone()
        block_length = block.shape[1]

        ### Observations ###
        if obs_history is None:
            raise ValueError("Block Gibbs smoothing requires obs_history.")

        obs_step_indices = list(range(block_length))
        obs_indices = [block_start + idx for idx in obs_step_indices]

        obs_block = torch.cat([
            self._flatten_obs(torch.as_tensor(obs_history[idx], device=device, dtype=torch.float32))
            for idx in obs_indices
        ], dim=1).to(device=device, dtype=torch.float32)

        obs_mask_window = None
        if not avg and obs_mask_history is not None:
            mask_shape = obs_mask_history[0].squeeze(0).shape
            obs_mask_window = torch.zeros(
                block_length,
                *mask_shape,
                dtype=torch.bool,
                device=device,
            )
            for global_idx in obs_indices:
                local_idx = global_idx - block_start
                obs_mask_window[local_idx] = obs_mask_history[global_idx].squeeze(0).to(device)

        self.model.observation_fn = self._make_block_obs_fn(
            obs_fn,
            scale,
            obs_mask_window=obs_mask_window,
            obs_step_indices=obs_step_indices,
            avg=avg,
        )
        self.model.obs_sigma = self.obs_sigma

        ### Schedule ###
        touches_left_boundary = block_start == 0
        touches_right_boundary = block_stop == n_times
        if touches_left_boundary and touches_right_boundary:
            left_tmin = self.gibbs_midpoint_tmin
            right_tmin = self.gibbs_midpoint_tmin
        elif touches_left_boundary:
            left_tmin = 1.0  # freezes x1, since x0 is not part of the trajectory
            right_tmin = self.gibbs_endpoint_tmin
        elif touches_right_boundary:
            left_tmin = self.gibbs_endpoint_tmin
            right_tmin = self.gibbs_midpoint_tmin
        else:
            left_tmin = self.gibbs_endpoint_tmin
            right_tmin = self.gibbs_endpoint_tmin

        step_tmin = self._build_step_tmin(
            block_length,
            mid_tmin=self.gibbs_midpoint_tmin,
            left_tmin=left_tmin,
            right_tmin=right_tmin,
            n_endpoints=self.gibbs_num_endpoints,
        )

        step_tmax = [1.0] * block_length

        print("tmin:", step_tmin)
        print("tmax:", step_tmax)
        if self.noise == "SDEdit":
            print("Using SDEdit")
            step_tmin_tensor = torch.tensor(
                step_tmin,
                device=device,
                dtype=block.dtype,
            ).view(1, block_length, 1, 1, 1)

        ### Steps 1 and 2, one ensemble batch at a time ###
        chunks = self._member_chunks(n_ens)
        if len(chunks) > 1:
            print(f"Gibbs block: {n_ens} members in {len(chunks)} batches of "
                  f"at most {self.ens_batch_size}")
        posterior = torch.empty_like(block)

        for start, stop in chunks:
            block_chunk = block[start:stop]
            # Shared observations broadcast; per-member ones (x0 guidance) are sliced.
            obs_chunk = obs_block[start:stop] if obs_block.shape[0] == n_ens else obs_block
            invert_schedule, forward_schedule = self._make_schedules(
                step_tmin,
                step_tmax,
                stop - start,
            )

            ### Step 1: Generate latents ###
            if self.noise == "SDEdit":
                zt = (step_tmin_tensor * block_chunk
                      + (1.0 - step_tmin_tensor) * torch.randn_like(block_chunk))
            else:
                with torch.no_grad():
                    zt = self.model.eta_stochastic_sampler(
                        block_chunk,
                        num_steps=self.invert_steps,
                        eps=self.invert_eps,
                        schedule=invert_schedule,
                        obs=None,
                    )
            ### Step 2: Generate posterior ###
            # The sampler returns a detached tensor, so each chunk's graph is freed.
            with torch.enable_grad():
                posterior[start:stop] = self.model.eta_stochastic_sampler(
                    zt,
                    num_steps=steps,
                    eps=self.eps,
                    schedule=forward_schedule,
                    obs=obs_chunk,
                )
            del zt

        updated = trajectory.clone()
        updated[:, block_start:block_stop] = posterior
        return updated

    def smooth(self, smooth_args, **kwargs):
        obs_history = smooth_args.obs_history
        obs_mask_history = smooth_args.obs_mask_history
        obs_fn = smooth_args.obs_fn
        n_times = smooth_args.n_times
        window_radius = smooth_args.window_radius
        steps = smooth_args.steps
        avg = smooth_args.avg
        trajectory_init = smooth_args.trajectory_init
        return_traj = getattr(smooth_args, "return_traj", False)
        num_sweeps = getattr(smooth_args, "num_sweeps", None)
        blocks = getattr(smooth_args, "blocks", None)
        block_mode = getattr(smooth_args, "block_mode", None)
        block_stride = getattr(smooth_args, "block_stride", None)
        block_direction = getattr(smooth_args, "block_direction", None)
        # Called with (sweep_idx, trajectory) after every sweep.
        sweep_callback = getattr(smooth_args, "sweep_callback", None)

        device = self.device
        n_ens = self.members
        steps = self.sampler_steps if steps is None else steps

        if window_radius in (None, 0):
            window_radius = self.block_radius
        expected_window = self.init_states + 1
        block_length = 2 * window_radius + 1
        if block_length != expected_window:
            raise ValueError(
                f"Block size {block_length} does not match model window size {expected_window}. "
                "Set init_states so that init_states + 1 equals the block length.")

        if trajectory_init is None:
            # State shape (channels, ny, nx) from the dataset metadata.
            trajectory = torch.randn((n_ens, n_times, *self.model.md.state_shape), device=device)
            print('Loader: Initialized random trajectory with shape', trajectory.shape)
        else:
            trajectory = torch.as_tensor(trajectory_init, device=device, dtype=torch.float32)

            if trajectory.dim() == 4:
                trajectory = trajectory.unsqueeze(0).repeat(n_ens, 1, 1, 1, 1)
            elif trajectory.dim() != 5:
                raise ValueError(
                    f"trajectory_init must have shape (T, D, X, Y) or (n_ens, T, D, X, Y), got {tuple(trajectory.shape)}")

        sweep_count = self.gibbs_sweeps if num_sweeps is None else num_sweeps
        for sweep_idx in range(sweep_count):
            sweep_centers = self._resolve_block_centers(
                n_times,
                window_radius,
                blocks=blocks,
                block_mode=block_mode,
                block_stride=block_stride,
            )
            if not sweep_centers:
                break

            # 'alternate' takes its parity from the sweep index.
            sweep_centers = self._ordered_centers(sweep_centers, block_direction=block_direction, sweep_count=sweep_idx)
            print('centers', sweep_centers)

            for center in tqdm(sweep_centers):
                trajectory = self.gibbs_step(
                    trajectory=trajectory,
                    block_start=center - window_radius,
                    block_stop=center + window_radius + 1,
                    obs_history=obs_history,
                    obs_mask_history=obs_mask_history,
                    obs_fn=obs_fn,
                    steps=steps,
                    avg=avg,
                )

            if sweep_callback is not None:
                sweep_callback(sweep_idx, trajectory)

        if return_traj:
            return trajectory
        return trajectory

    def assimilate(self, smooth_args, **kwargs):
        return self.smooth(smooth_args, **kwargs)