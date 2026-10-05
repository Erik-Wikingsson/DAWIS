import torch

from assimilation.methods.method import AssimilationMethod
from forecasting.ckpt_args import (
    apply_checkpoint_arch,
    arch_from_checkpoint,
    check_nx,
    read_checkpoint,
)
from forecasting.models.fmw import FMW, LinearAlphaBetaSchedule


def _setdefault(args, key, value):
    if not hasattr(args, key) or getattr(args, key) is None:
        setattr(args, key, value)


class DAWIS(AssimilationMethod):
    """
    DAWIS: Data Assimilation with Windowed Inverse Sampling via Multitask
    Interpolants.

    Assimilates with a pretrained windowed flow-matching model (FMW) over a
    window of time steps; DAISI is the single-state special case.

    The FMW state is kept in normalized units (x / scale). The scale is
    composed into the observation operator, so guidance is computed in the
    observer's (scalefact) units.
    """

    def __init__(self, args, metadata=None):
        # `--model_path` if given, else the dataset entry in assimilation/checkpoints.py.
        fm_loss = getattr(args, "fm_loss", "eta01_label")
        if metadata is not None:
            from assimilation.checkpoints import resolve_window_model
            args.model_path = resolve_window_model(metadata, args)
        elif args.model_path in (None, "", "None"):
            raise ValueError(
                "DAWIS needs either --model_path or the dataset metadata to "
                "resolve its FMW checkpoint."
            )
        print(
            f"DAWIS resolving fm_loss='{fm_loss}' to checkpoint {args.model_path}")

        # Inference defaults; the architecture comes from the checkpoint.
        _setdefault(args, "sampler", "stochastic")
        _setdefault(args, "sampler_steps", 100)
        _setdefault(args, "sampler_eps", 0)
        _setdefault(args, "corr_noise", None)
        _setdefault(args, "time_dropout", None)
        _setdefault(args, "guide_method", "DPS")
        from forecasting.models.ar_model import apply_inference_defaults
        apply_inference_defaults(args)

        super().__init__(args)
        self.args = args

        device = args.device
        self.device = device

        info = read_checkpoint(args.model_path, map_location=device)
        if not info.is_lightning:
            raise ValueError(
                f"DAWIS needs an FMW window model (a Lightning .ckpt saved by "
                f"forecasting/trainer.py), but {args.model_path} is a bare "
                f"state_dict. The DAISI priors look like this -- see "
                f"assimilation/checkpoints.py for which checkpoint each method "
                f"wants.")
        if info.model is not None and info.model.upper() != "FMW":
            raise ValueError(
                f"DAWIS needs an FMW window model, but {args.model_path} was "
                f"trained as model={info.model}. Point --model_path at an FMW "
                f"checkpoint (see FMW_MODELS in assimilation/checkpoints.py).")

        arch = arch_from_checkpoint(info)
        check_nx(arch, metadata, label="DAWIS")
        # `--init_states` also sizes --tmin/--tmax and the saved window, so it
        # must match the checkpoint rather than be taken from it.
        ckpt_init = arch.get("init_states")
        if ckpt_init is not None and int(ckpt_init) != int(args.init_states):
            raise ValueError(
                f"--init_states {args.init_states} but {args.model_path} was "
                f"trained with init_states={ckpt_init}.\n"
                f"  --init_states also sizes --tmin/--tmax/--tmin_init (length "
                f"{int(ckpt_init) + 1}) and the saved window, which are already "
                f"fixed by now, so it cannot be taken from the checkpoint.\n"
                f"  Pass --init_states {ckpt_init}, or point --model_path at the "
                f"init_states={args.init_states} checkpoint.")

        # Build the model from a copy of args carrying the checkpoint's architecture.
        from assimilation.parser import arch_defaults
        model_args = apply_checkpoint_arch(
            args, info, label="DAWIS", defaults=arch_defaults())

        self.model = FMW(model_args).to(device)
        self.model.load_state_dict(info.state_dict)
        self.model.eval()
        self.arch = arch

        # Divides the assimilation state into network units: data_std * unit_factor
        # for physical-unit data (SQG), 1 when the data is already in model space (SEVIR).
        self.scale = self.model.md.assim_scale(self.model.data_std)

        self.members = getattr(args, "n_ens", 20)
        self.init_states = args.init_states  # FMW window width (history steps)

        # Per-state inversion depth, length T = init_states + 1, ordered
        # [cond_0, ..., cond_{window-1}, target]:
        #   tmin[i] == 1.0 -> held at clean data (beta = 1), no inversion.
        #   tmin[i]  < 1.0 -> inverted 1 -> tmin[i], then sampled tmin[i] -> tmax[i].
        self.tmin = self._validate_tmin(
            getattr(args, "tmin", None), args.init_states + 1)

        # Per-state depth where the guided forward phase ends (default 1.0 = clean).
        # tmax[i] < 1 leaves state i partially noised for later cycles (rolling
        # "pyramid" schedule); tmax[i] == tmin[i] freezes it.
        self.tmax = self._validate_tmax(
            getattr(args, "tmax", None), self.tmin, args.init_states + 1)

        # Initialization-phase override for the conditioning slots. See `_step_tmin`.
        self.tmin_init = self._validate_tmin_init(
            getattr(args, "tmin_init", None), args.init_states)

        # forward_model='dawis': the target slot is generated from pure noise
        # (beta 0 -> tmin[-1]) in stage 1 instead of inverted from a forecast.
        self.generate_target = getattr(args, "forward_model", None) == "dawis"
        if self.generate_target and self.tmin[-1] >= 1.0:
            raise ValueError(
                "forward_model='dawis' generates the target step, so tmin[-1] must "
                f"be < 1.0; got {self.tmin[-1]}.")

        # Apply observation guidance during the FMW prediction step (forward_model='fmw').
        self.guide_forecast = getattr(args, "guide_forecast", False)
        self.guide_first = getattr(args, "guide_first", "none")
        # Stage-1 guidance is opt-in via --guide_stage1 (store_true); this
        # default only applies to a hand-built args namespace.
        self.guide_stage1 = getattr(args, "guide_stage1", True)
        self.invert_steps = getattr(args, "invert_steps", 100)
        self.invert_eps = getattr(args, "invert_eps", 0.0)
        self.sampler_steps = getattr(args, "sampler_steps", 100)
        self.noise = getattr(args, "noise", "invert")
        self.obs_sigma = getattr(args, "obs_sigma", 1.0)
        self.guide_all_steps = getattr(args, "guide_all_steps", False)
        self.guidance_strength = getattr(args, "guidance_strength", 1.0)

        # eps for the forward SDE (no scale² factor: state is normalized, unlike DAISI)
        _eps_base = getattr(args, "eps", 0.0)
        self.eps = _eps_base

        print(f"DAWIS initialized | device={device} | "
              f"fm_loss={getattr(self.model.args, 'fm_loss', fm_loss)} | "
              f"guide={args.guide_method} | "
              f"tmin={self.tmin} | tmax={self.tmax} | tmin_init={self.tmin_init} | "
              f"guide_forecast={self.guide_forecast} | guide_first={self.guide_first} | noise={self.noise} | "
              f"generate_target={self.generate_target}")
        if self.generate_target and self.noise != "invert":
            print("DAWIS: forward_model='dawis' generates the target from pure noise; "
                  f"--noise '{self.noise}' is ignored for the target slot.")

    def set_guide_method(self, guide_method):
        self.model.set_guide_method(guide_method)

    def set_guidance_strength(self, strength):
        self.model.guidance_strength = strength

    @staticmethod
    def _validate_tmin(tmin, expected_len):
        """Validate per-state `tmin` (length init_states + 1, target last).

        None falls back to 0.0 everywhere with a warning; a scalar is rejected.
        """
        if tmin is None:
            tmin = [0.0] * (expected_len)
            print(
                "DAWIS requires `tmin` as a per-state list of length "
                f"T = init_states + 1 = {expected_len}; got None.")
        if isinstance(tmin, (int, float)):
            raise ValueError(
                "DAWIS requires `tmin` as a per-state list of length "
                f"T = init_states + 1 = {expected_len}; got scalar {tmin}. "
                "Pass e.g. tmin=[1.0, 0.3] (conditioning first, target last).")
        if isinstance(tmin, torch.Tensor):
            tmin = tmin.flatten().tolist()
        else:
            tmin = list(tmin)
        if len(tmin) != expected_len:
            raise ValueError(
                f"DAWIS `tmin` has length {len(tmin)} but expected "
                f"T = init_states + 1 = {expected_len}.")
        return [float(t) for t in tmin]

    @staticmethod
    def _validate_tmax(tmax, tmin, expected_len):
        """Validate per-state `tmax` (same layout as `tmin`).

        Requires tmin[i] <= tmax[i] <= 1. None means 1.0 everywhere.
        """
        if tmax is None:
            return [1.0] * expected_len
        if isinstance(tmax, (int, float)):
            raise ValueError(
                "DAWIS requires `tmax` as a per-state list of length "
                f"T = init_states + 1 = {expected_len}; got scalar {tmax}. "
                "Pass e.g. tmax=[1.0, 0.5] (conditioning first, target last).")
        if isinstance(tmax, torch.Tensor):
            tmax = tmax.flatten().tolist()
        else:
            tmax = list(tmax)
        if len(tmax) != expected_len:
            raise ValueError(
                f"DAWIS `tmax` has length {len(tmax)} but expected "
                f"T = init_states + 1 = {expected_len}.")
        tmax = [float(t) for t in tmax]
        if not all(0.0 <= t <= 1.0 for t in tmax):
            raise ValueError(
                f"DAWIS `tmax` entries must be in [0, 1]; got {tmax}.")
        for i, (lo, hi) in enumerate(zip(tmin, tmax)):
            if hi < float(lo) - 1e-8:
                raise ValueError(
                    f"DAWIS `tmax[{i}]` = {hi} is below `tmin[{i}]` = {float(lo)}: "
                    "the guided forward phase samples beta from tmin up to tmax, so "
                    "tmax must be >= tmin (equal freezes the state at that level).")
        return tmax

    @staticmethod
    def _validate_tmin_init(tmin_init, expected_len):
        """Validate `tmin_init` (conditioning slots only, length init_states).

        None disables the override.
        """
        if tmin_init is None:
            return None
        if isinstance(tmin_init, (int, float)):
            raise ValueError(
                "DAWIS `tmin_init` must be a per-slot list of length "
                f"init_states = {expected_len}; got scalar {tmin_init}. "
                "Pass e.g. tmin_init=[0.3, 0.3, 1.0] (oldest slot first, the slot "
                "holding the initial condition last).")
        if isinstance(tmin_init, torch.Tensor):
            tmin_init = tmin_init.flatten().tolist()
        else:
            tmin_init = list(tmin_init)
        if len(tmin_init) != expected_len:
            raise ValueError(
                f"DAWIS `tmin_init` has length {len(tmin_init)} but expected "
                f"init_states = {expected_len} (conditioning slots only; the target "
                "keeps tmin[-1]).")
        tmin_init = [float(t) for t in tmin_init]
        if not all(0.0 <= t <= 1.0 for t in tmin_init):
            raise ValueError(
                f"DAWIS `tmin_init` entries must be in [0, 1]; got {tmin_init}.")
        return tmin_init

    def _step_tmin(self, ntime):
        """Per-state inversion depths at step `ntime` (length init_states + 1).

        `self.tmin`, except that while the initial condition is still in the
        window, the leading slots are overridden by `tmin_init[ntime:]`. Its
        last entry then lands on the initial-condition slot (window - 1 - ntime).
        Derived from `ntime` (not popped) since `assimilate` may be called once
        per ensemble batch within a step.
        """
        step_tmin = [float(t) for t in self.tmin]
        if self.tmin_init is not None:
            for j, t in enumerate(self.tmin_init[int(ntime):]):
                step_tmin[j] = t
        return step_tmin

    def _make_schedules(self, step_tmin, step_tmax, n_ens, gen_flags=None):
        """Build the (n_ens, T) stage-1 and stage-2 LinearAlphaBetaSchedules.

        Per state i:
          - step_tmin[i] >= 1: held at clean data (beta = 1) in both stages.
          - otherwise stage 1 goes beta 1 -> tmin[i] (invert) or, if gen_flags[i],
            0 -> tmin[i] (generate from noise); stage 2 goes tmin[i] -> tmax[i].

        Returns:
            (invert_schedule, forward_schedule)
        """
        device = self.device
        T = len(step_tmin)
        if gen_flags is None:
            gen_flags = [False] * T

        def _build(forward):
            tstart, tend = [], []
            for i in range(T):
                ti = step_tmin[i]
                if ti >= 1.0:
                    tstart.append(1.0)
                    tend.append(1.0)
                elif forward:
                    tstart.append(ti)
                    tend.append(step_tmax[i])
                else:
                    tstart.append(0.0 if gen_flags[i] else 1.0)
                    tend.append(ti)
            tstart = torch.tensor([tstart], device=device).repeat(n_ens, 1)
            tend = torch.tensor([tend], device=device).repeat(n_ens, 1)
            return LinearAlphaBetaSchedule(tstart=tstart, tend=tend)

        return _build(forward=False), _build(forward=True)

    def assimilate(self, x_forecast, init_states, obs, obs_mask, obs_fn,
                   ntime=0, debug=False, return_traj=False, invert=True, **kwargs):
        """
        Perform one data assimilation step using the windowed FMW model.

        Args:
            x_forecast: Forecast/prior for the *target* step,
                shape (n_ens, D, X, Y) in scalefact units.
                Pass None for pure-noise initialization.
            init_states: Conditioning window, shape (window, n_ens, D, X, Y)
                in scalefact units. window == self.init_states.
            obs: Observations in scalefact units, shape (n_obs,) or (2, n_obs).
            obs_mask: Boolean mask, shape (1, D, X, Y).
            obs_fn: Observation operator applied to the current observation in scalefact units.
            ntime: Current assimilation time index (selects `tmin_init` slots).
            debug: If True, return (analysis, latents_window, posterior_window).
            return_traj: If True, also return the posterior conditioning window.
            **kwargs: obs_history, obs_mask_history, obs_fn_history,
                obs_sigma_history (used with guide_all_steps).

        Returns:
            x_analysis: Assimilated state, shape (n_ens, D, X, Y) in scalefact units.
        """
        scale = self.scale
        device = self.device
        # (window, obs_dim) if guide_all_steps
        obs_history = kwargs.get("obs_history", None)
        obs_mask_history = kwargs.get(
            "obs_mask_history", None)  # (window, 1, D, X, Y)
        obs_fn_history = kwargs.get("obs_fn_history", None)
        obs_sigma_history = kwargs.get("obs_sigma_history", None)

        # 1. Normalize inputs to FMW units
        if init_states is None:
            init_states = torch.randn((self.args.init_states,
                                       self.args.n_ens,
                                       *self.model.md.state_shape),
                                      device=device) * scale

        if not isinstance(init_states, torch.Tensor):
            init_states = torch.tensor(
                init_states, device=device, dtype=torch.float32)
        else:
            init_states = init_states.to(device=device, dtype=torch.float32)

        # init_states: (window, n_ens, D, X, Y) → (n_ens, window, D, X, Y)
        cond_norm = (init_states / scale).permute(1, 0, 2, 3, 4)
        n_ens, window, D, X, Y = cond_norm.shape

        # Per-state depths for this step (needed by SDEdit seeding below).
        T = window + 1
        step_tmin = self._step_tmin(ntime)
        assert len(step_tmin) == T, (
            f"tmin length {len(step_tmin)} != T={T} (init_states + 1)")
        step_tmax = [float(t) for t in self.tmax]
        assert len(step_tmax) == T, (
            f"tmax length {len(step_tmax)} != T={T} (init_states + 1)")

        if x_forecast is not None:
            if not isinstance(x_forecast, torch.Tensor):
                x_forecast = torch.tensor(
                    x_forecast, device=device, dtype=torch.float32)
            else:
                x_forecast = x_forecast.to(device=device, dtype=torch.float32)
            if torch.isnan(x_forecast).any():
                # Diverged forecast: skip the update, return the window unchanged.
                print("DAWIS: prior contains NaNs — skipping assimilation.")
                if return_traj:
                    return x_forecast, init_states.permute(1, 0, 2, 3, 4)
                return x_forecast
            target_norm = x_forecast / scale  # (n_ens, D, X, Y)
        else:
            target_norm = None

        # 2. Build latents (n_ens, T, D, X, Y): conditioning window + target slot
        target_latent = self._build_target_noise(
            target_norm, n_ens, D, X, Y, tmin_target=step_tmin[-1])
        latents = torch.cat([cond_norm, target_latent.unsqueeze(1)], dim=1)

        # 3. Per-state schedules (see `_make_schedules`)
        print(f"DAWIS step ntime={ntime} | beta "
              f"{['%.3f->%.3f' % (a, b) for a, b in zip(step_tmin, step_tmax)]}",
              flush=True)
        invert_flags = [t < 1.0 for t in step_tmin]
        # Only the target slot can be generated from noise.
        gen_flags = [False] * T
        if self.generate_target:
            gen_flags[-1] = True

        invert_schedule, forward_schedule = self._make_schedules(
            step_tmin, step_tmax, n_ens, gen_flags=gen_flags)

        # 4. Observation operator (shared by both stages), in scalefact units
        obs_mask_window = self._build_obs_mask_window(
            obs_mask, T, D, X, Y,
            obs_mask_history=obs_mask_history if self.guide_all_steps else None
        )
        print("observed", [obs_mask_window[k].flatten().any().item() for k in range(obs_mask_window.shape[0])], flush=True)

        if self.guide_all_steps and obs_history is not None:
            obs_fn_window = list(obs_fn_history) if obs_fn_history is not None else [obs_fn] * len(obs_history)
            obs_fn_window.append(obs_fn)

            self.model.observation_fn = self._make_history_obs_fn(
                obs_fn_window,
                scale,
                obs_mask_window,
            )
            obs_guided = self._build_windowed_obs(obs, obs_history)
            obs_sigma_guided = self._build_windowed_obs_sigma(
                obs,
                obs_history,
                obs_sigma_history,
                self.obs_sigma,
            )
        else:
            self.model.observation_fn = self._make_masked_obs_fn(
                obs_fn, scale, obs_mask_window)

            obs_guided = obs

            obs_sigma_guided = torch.as_tensor(self.obs_sigma, device=device, dtype=torch.float32)

        self.model.obs_sigma = obs_sigma_guided

        # 5. Stage 1: invert data-backed slots and/or generate the target
        run_stage1 = any(gen_flags) or (
            any(invert_flags) and self.noise == "invert")
        # Stage-1 guidance only steers generated slots; inverted slots are masked out.
        guide_stage1 = self.guide_stage1 and any(gen_flags) and obs_guided is not None

        if run_stage1:
            base_guidance_strength = self.model.guidance_strength
            try:
                if guide_stage1:
                    # (1, T) mask broadcast against FMW's (B, T) guidance strength.
                    gen_mask = torch.tensor(
                        [[1.0 if g else 0.0 for g in gen_flags]], device=device)
                    self.model.guidance_strength = (
                        lambda t: base_guidance_strength(t) * gen_mask)
                    with torch.enable_grad():
                        zt = self.model.eta_stochastic_sampler(
                            latents,
                            num_steps=self.invert_steps,
                            eps=self.invert_eps,
                            schedule=invert_schedule,
                            obs=obs_guided,
                        )
                else:
                    with torch.no_grad():
                        zt = self.model.eta_stochastic_sampler(
                            latents,
                            num_steps=self.invert_steps,
                            eps=self.invert_eps,
                            schedule=invert_schedule,
                            obs=None,
                        )
            finally:
                self.model.guidance_strength = base_guidance_strength
        else:
            # Nothing to invert: fill beta = 0 slots with fresh noise.
            pure_noise = [t==0.0 for t in step_tmin]
            latents[:, pure_noise] = torch.randn_like(latents[:, pure_noise])
            zt = latents

        # 6. Stage 2: guided forward phase over all slots
        with torch.enable_grad():
            posterior = self.model.eta_stochastic_sampler(
                zt,
                num_steps=self.sampler_steps,
                eps=self.eps,
                schedule=forward_schedule,
                obs=obs_guided,
            )

        # 7. Analysis in scalefact units
        analysis = posterior[:, -1] * scale  # (n_ens, D, X, Y)

        if return_traj:
            return analysis, posterior[:, :-1] * scale.unsqueeze(0)

        if debug:
            # Post-guidance observation misfit, in scalefact units.
            with torch.no_grad():
                y_hat = self.model.observation_fn(
                    posterior)  # (n_ens, n_obs_total)
                misfit = obs_guided - y_hat
                rms_misfit = torch.sqrt((misfit ** 2).mean()).item()
            print(f"DAWIS debug | post-guidance obs misfit RMS={rms_misfit:.4f} "
                  f"(scalefact units) | obs={tuple(obs_guided.shape)} "
                  f"y_hat={tuple(y_hat.shape)}")
            return analysis, zt, posterior
        return analysis

    def forecast(self, init_states, obs=None, obs_mask=None, obs_fn=None):
        """
        Forecast the next state with the FMW model (predict_step), optionally
        guided by observations valid at the forecast time.

        Args:
            init_states: Analysis history window, shape (window, n_ens, D, X, Y)
                in scalefact units.
            obs: Observations at the forecast time, in scalefact units. Only used
                when self.guide_forecast is True (otherwise a pure forecast).
            obs_mask: Boolean mask, shape (1, D, X, Y), for the forecast-time obs.
            obs_fn: Observation operator applied to state in scalefact units.

        Returns:
            x_forecast: (n_ens, D, X, Y) in scalefact units.
        """
        scale = self.scale
        device = self.device
        if not isinstance(init_states, torch.Tensor):
            init_states = torch.tensor(
                init_states, device=device, dtype=torch.float32)
        else:
            init_states = init_states.to(device=device, dtype=torch.float32)
        # (window, n_ens, D, X, Y) -> (n_ens, window, D, X, Y)
        cond_norm = (init_states / scale).permute(1, 0, 2, 3, 4)

        if self.guide_forecast and obs is not None and obs_fn is not None:
            n_ens, window, D, X, Y = cond_norm.shape
            T = window + 1  # conditioning window + target
            # Only the target (last) step carries the forecast-time observation.
            mask_window = self._build_obs_mask_window(obs_mask, T, D, X, Y)
            self.model.observation_fn = self._make_masked_obs_fn(
                obs_fn, scale, mask_window)
            self.model.obs_sigma = self.obs_sigma
            with torch.enable_grad():
                pred_norm = self.model.predict_step(
                    cond_norm, obs=obs.to(device))
        else:
            with torch.no_grad():
                pred_norm = self.model.predict_step(cond_norm)

        return pred_norm * scale  # (n_ens, D, X, Y)

    def _build_target_noise(self, target_norm, n_ens, D, X, Y, tmin_target=None):
        """Initial latent for the target slot, in normalized units.

        noise='invert' clones the forecast (inverted in stage 1), 'SDEdit' blends
        it with noise at `tmin_target` (default `self.tmin[-1]`), a tensor is used
        as-is, and None gives fresh noise. With forward_model='dawis' the slot is
        always fresh noise.
        """
        device = self.device
        noise = self.noise

        if self.generate_target:
            return torch.randn(n_ens, D, X, Y, device=device)

        # An explicit latent takes precedence, even without a forecast
        # (e.g. the pyramid spin-down in assimilate.py).
        if isinstance(noise, torch.Tensor):
            expected = (n_ens, D, X, Y)
            if noise.shape != expected:
                raise ValueError(
                    f"Noise tensor shape {noise.shape} != expected {expected}")
            return noise.to(device=device, dtype=torch.float32)

        if target_norm is None or noise is None or noise == "None":
            return torch.randn(n_ens, D, X, Y, device=device)

        if noise == "invert":
            return target_norm.clone()

        if noise == "SDEdit":
            # Linear schedule: alpha(tmin) = 1 - tmin, beta(tmin) = tmin.
            if tmin_target is None:
                tmin_target = self.tmin[-1]
            alpha_tmin = 1.0 - tmin_target
            beta_tmin = tmin_target
            return beta_tmin * target_norm + alpha_tmin * torch.randn_like(target_norm)

        raise ValueError(
            f"Unsupported noise='{noise}'. Must be 'invert', 'SDEdit', None, or a Tensor.")

    def _build_obs_mask_window(self, obs_mask, T, D, X, Y, obs_mask_history=None):
        """
        Build a (T, D, X, Y) boolean mask.
        - guide_all_steps=False: only the last time step is observed.
        - guide_all_steps=True: use obs_mask_history for earlier steps + obs_mask for last.
        """
        device = self.device
        mask_window = torch.zeros(T, D, X, Y, dtype=torch.bool, device=device)

        if self.guide_all_steps and obs_mask_history is not None:
            # obs_mask_history: (window, 1, D, X, Y)
            n_obs = min(len(obs_mask_history), T - 1)
            t_start = (T - 1) - n_obs
            for t in range(n_obs):
                mask_window[t + t_start] = obs_mask_history[t].squeeze(0).to(device)

        # guide_first='always': fully observe the first slot.
        if self.guide_first=="always":
            n_obs = min(len(obs_mask_history) if obs_mask_history is not None else 0, T - 1)
            t_start = min((T - 1) - n_obs - 1, 0)
            mask_window[t_start] = torch.full_like(mask_window[0], True)

        # Last time step always observed (current obs)
        mask_window[-1] = obs_mask.squeeze(0).to(device)
        return mask_window

    def _make_masked_obs_fn(self, obs_fn, scale, obs_mask_window):
        """
        Returns an observation function compatible with FMW's euler_step.
        FMW passes z_norm (B, T, D, X, Y); we map to scalefact units before obs_fn.
        obs_mask_window: (T, D, X, Y) bool.
        """
        def _obs_fn(z_norm):
            x_sf = z_norm * scale  # to scalefact units
            # Select masked (T, D, X, Y) elements per member -> (B, n_obs_total)
            if self.guide_first=="always":
                obs_init = x_sf[:, 0, obs_mask_window[0]]
                obs_rest = obs_fn(x_sf[:, 1:][:, obs_mask_window[1:]])
                return torch.cat([obs_init, obs_rest], dim=1)

            observed = x_sf[:, obs_mask_window]
            return obs_fn(observed)
        return _obs_fn

    def _make_avg_obs_fn(self, obs_fn, scale):
        """
        AvgObserver: obs_fn takes full (B, D, X, Y) state at the *last* time step.
        """
        def _obs_fn(z_norm):
            # z_norm: (B, T, D, X, Y) — use last step
            x_sf = z_norm[:, -1] * scale  # (B, D, X, Y) in scalefact units
            return obs_fn(x_sf)
        return _obs_fn

    def _build_windowed_obs(self, obs_current, obs_history):
        """
        Concatenate obs_history + obs_current for guide_all_steps mode.
        obs_history: list/tensor of past obs vectors.
        obs_current: current obs vector.
        Returns concatenated obs tensor (device-agnostic; FMW will see it).
        """
        if obs_history is None:
            return obs_current
        obs_list = list(obs_history) + [obs_current]
        # Shared obs are (1, n_obs), per-member ones (n_ens, n_obs); broadcast to the larger.
        n_ens = max(o.shape[0] for o in obs_list)
        new_list = torch.cat([o.expand(n_ens, -1) for o in obs_list], dim=1)
        return new_list

    def _build_windowed_obs_sigma(self, obs_current, obs_history, obs_sigma_history, obs_sigma_current):
        """
        Build a concatenated observation-noise tensor aligned with the windowed
        observation vector.
        """
        sigma_list = []
        history = list(obs_history) if obs_history is not None else []
        sigma_history = list(obs_sigma_history) if obs_sigma_history is not None else []

        for obs_item, sigma_item in zip(history, sigma_history):
            sigma_tensor = torch.as_tensor(sigma_item, device=self.device, dtype=torch.float32)
            if sigma_tensor.ndim == 0:
                sigma_tensor = sigma_tensor.expand(obs_item[0].flatten().shape[0])
            else:
                sigma_tensor = sigma_tensor.flatten()
            sigma_list.append(sigma_tensor)

        current_sigma = torch.as_tensor(obs_sigma_current, device=self.device, dtype=torch.float32)
        if current_sigma.ndim == 0:
            current_sigma = current_sigma.expand(obs_current[0].flatten().shape[0])
        else:
            current_sigma = current_sigma.flatten()
        sigma_list.append(current_sigma)

        return torch.cat(sigma_list, dim=0).unsqueeze(0)  # (1, n_obs_total)

    def _make_history_obs_fn(self, obs_fn_history, scale, obs_mask_window):
        """
        Returns an observation function for a window of states, applying each
        time step's own observation operator to its masked state.
        """
        def _obs_fn(z_norm):
            x_sf = z_norm * scale
            obs_parts = []

            T = x_sf.shape[1]
            n_obs = min(len(obs_fn_history) if obs_fn_history is not None else 0, T)
            t_start = T - n_obs
            for t in range(n_obs):
                obs_fn = obs_fn_history[t]
                state = x_sf[:, t+t_start]
                obs_masked = state[:, obs_mask_window[t+t_start]]
                obs_step = obs_fn(obs_masked)
                obs_parts.append(obs_step.reshape(obs_step.shape[0], -1))

            return torch.cat(obs_parts, dim=1)

        return _obs_fn
