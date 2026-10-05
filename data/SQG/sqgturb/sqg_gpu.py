"""Differentiable, batched PyTorch implementation of the SQG model in `sqg.py`.

Same formulation (pseudo-spectral, dealiased Jacobian, RK4 with integrating-
factor hyperdiffusion) using `torch.fft`, so gradients are available by
backpropagation (cf. Solvik et al., 2025, doi:10.1029/2024MS004608). State is
(n_ens, 2, N, N); a bare (2, N, N) input is also accepted. `advance(pv)` does
not mutate internal state.
"""
import numpy as np
import torch


class SQGTorch:
    """Pseudo-spectral SQG model in PyTorch.

    Args and attribute names mirror `sqg.SQG` so the two can be swapped.

    Args:
        pv: initial PV, shape (2, N, N) or (n_ens, 2, N, N).
        threads: accepted and ignored (pyfftw-only knob in `sqg.SQG`).
        device: torch device for the model constants and state.
    """

    def __init__(
        self,
        pv,
        f=1.0e-4,
        nsq=1.0e-4,
        L=20.0e6,
        H=10.0e3,
        U=30.0,
        r=0.0,
        tdiab=10.0 * 86400,
        diff_order=8,
        diff_efold=None,
        symmetric=True,
        dt=None,
        dealias=True,
        threads=1,
        precision="double",
        tstart=0,
        device=None,
    ):
        pv = torch.as_tensor(np.asarray(pv))
        if pv.dim() == 3:
            pv = pv.unsqueeze(0)
        if pv.shape[1] != 2:
            raise ValueError("2nd dim of pv should be 2 (got shape %s)" % (tuple(pv.shape),))
        N = pv.shape[-1]
        if N % 2:
            raise ValueError("N must be even (powers of 2 are fastest)")
        if dt is None:
            raise ValueError("must specify time step")
        if diff_efold is None:
            raise ValueError("must specify efolding time scale for diffusion")

        if precision == "single":
            dtype = torch.float32
            cdtype = torch.complex64
        elif precision == "double":
            dtype = torch.float64
            cdtype = torch.complex128
        else:
            raise ValueError("precision must be 'single' or 'double'")

        self.device = torch.device(device) if device is not None else pv.device
        self.dtype = dtype
        self.cdtype = cdtype
        self.threads = threads
        self.N = N
        self.dealias = dealias
        self.symmetric = symmetric
        self.ekman = r >= 1.0e-10
        self.t = tstart
        self.timesteps = 1

        def _c(v):
            return torch.tensor(float(v), dtype=dtype, device=self.device)

        self.nsq, self.f, self.H = _c(nsq), _c(f), _c(H)
        self.U, self.L, self.dt = _c(U), _c(L), _c(dt)
        self.r, self.tdiab = _c(r), _c(tdiab)
        self.diff_order, self.diff_efold = _c(diff_order), _c(diff_efold)

        # --- basic state PV for thermal relaxation (as in sqg.py) ---
        y = np.arange(0, L, L / N, dtype=np.float64)
        pvbar = np.zeros((2, N), np.float64)
        pi = np.pi
        l0 = 2.0 * pi / L
        mu0 = l0 * np.sqrt(nsq) * H / f
        if symmetric:
            pvbar[:] = (-(mu0 * 0.5 * U / (l0 * H)) * np.cosh(0.5 * mu0)
                        * np.cos(l0 * y) / np.sinh(0.5 * mu0))
        else:
            pvbar[:] = -(mu0 * U / (l0 * H)) * np.cos(l0 * y) / np.sinh(mu0)
            pvbar[1, :] = pvbar[0, :] * np.cosh(mu0)
        pvbar = pvbar.reshape(2, N, 1) * np.ones((2, N, N), np.float64)
        self.pvbar = torch.as_tensor(pvbar, dtype=dtype, device=self.device)
        # (1, 2, N, N//2+1) so it broadcasts over the ensemble dim
        self.pvspec_eq = torch.fft.rfft2(self.pvbar).unsqueeze(0).to(cdtype)

        self.pvspec = torch.fft.rfft2(pv.to(dtype).to(self.device)).to(cdtype)

        # --- spectral grids ---
        k = (N * np.fft.fftfreq(N))[0:(N // 2) + 1]
        l = N * np.fft.fftfreq(N)
        k, l = np.meshgrid(k, l)
        k = 2.0 * pi * k / L
        l = 2.0 * pi * l / L
        ksqlsq = k ** 2 + l ** 2
        self.k = torch.as_tensor(k, dtype=dtype, device=self.device)
        self.l = torch.as_tensor(l, dtype=dtype, device=self.device)
        self.ksqlsq = torch.as_tensor(ksqlsq, dtype=dtype, device=self.device)
        self.ik = (1.0j * self.k).to(cdtype)
        self.il = (1.0j * self.l).to(cdtype)

        if dealias:
            k_pad = ((3 * N // 2) * np.fft.fftfreq(3 * N // 2))[0:(3 * N // 4) + 1]
            l_pad = (3 * N // 2) * np.fft.fftfreq(3 * N // 2)
            k_pad, l_pad = np.meshgrid(k_pad, l_pad)
            k_pad = 2.0 * pi * k_pad / L
            l_pad = 2.0 * pi * l_pad / L
            self.ik_pad = (1.0j * torch.as_tensor(
                k_pad, dtype=dtype, device=self.device)).to(cdtype)
            self.il_pad = (1.0j * torch.as_tensor(
                l_pad, dtype=dtype, device=self.device)).to(cdtype)

        # --- inversion constants ---
        mu = np.sqrt(ksqlsq) * np.sqrt(nsq) * H / f
        # Clip with the model dtype's epsilon, as sqg.py does, to avoid
        # float32 overflow at the (0, 0) mode.
        np_dtype = np.float32 if precision == "single" else np.float64
        mu = mu.astype(np_dtype).clip(np.finfo(np_dtype).eps).astype(np.float64)
        self.Hovermu = torch.as_tensor(H / mu, dtype=dtype, device=self.device)
        self.tanhmu = torch.as_tensor(np.tanh(mu), dtype=dtype, device=self.device)
        self.sinhmu = torch.as_tensor(np.sinh(mu), dtype=dtype, device=self.device)

        # --- integrating factor for hyperdiffusion ---
        ktot = np.sqrt(ksqlsq)
        ktotcutoff = pi * N / L
        hyperdiff = np.exp((-float(dt) / float(diff_efold))
                           * (ktot / ktotcutoff) ** float(diff_order))
        self.hyperdiff = torch.as_tensor(hyperdiff, dtype=dtype, device=self.device)

    # inversion
    def invert(self, pvspec=None):
        """Boundary PV -> streamfunction (spectral)."""
        if pvspec is None:
            pvspec = self.pvspec
        psi0 = self.Hovermu * ((pvspec[:, 1] / self.sinhmu) - (pvspec[:, 0] / self.tanhmu))
        psi1 = self.Hovermu * ((pvspec[:, 1] / self.tanhmu) - (pvspec[:, 0] / self.sinhmu))
        return torch.stack((psi0, psi1), dim=1)

    def invert_inverse(self, psispec=None):
        """Streamfunction -> PV (spectral)."""
        if psispec is None:
            psispec = self.invert(self.pvspec)
        alpha, th, sh = self.Hovermu, self.tanhmu, self.sinhmu
        tmp1 = 1.0 / sh ** 2 - 1.0 / th ** 2
        tmp1 = tmp1.clone()
        tmp1[0, 0] = 1.0
        pv0 = ((psispec[:, 0] / th) - (psispec[:, 1] / sh)) / (alpha * tmp1)
        pv1 = ((psispec[:, 0] / sh) - (psispec[:, 1] / th)) / (alpha * tmp1)
        pvspec = torch.stack((pv0, pv1), dim=1)
        # area mean PV is not determined by the streamfunction
        mask = torch.ones_like(pvspec.real)
        mask[:, :, 0, 0] = 0.0
        return pvspec * mask

    # dealiasing helpers
    def specpad(self, specarr):
        """Zero-pad spectral coefficients onto the 3/2 grid (2/3 rule)."""
        N = self.N
        B = specarr.shape[0]
        pad = specarr.new_zeros((B, 2, 3 * N // 2, 3 * N // 4 + 1))
        pad[:, :, 0:N // 2, 0:N // 2] = 2.25 * specarr[:, :, 0:N // 2, 0:N // 2]
        pad[:, :, -N // 2:, 0:N // 2] = 2.25 * specarr[:, :, -N // 2:, 0:N // 2]
        # include negative Nyquist frequency
        pad[:, :, 0:N // 2, N // 2] = torch.conj(2.25 * specarr[:, :, 0:N // 2, -1])
        pad[:, :, -N // 2:, N // 2] = torch.conj(2.25 * specarr[:, :, -N // 2:, -1])
        return pad

    def spectrunc(self, specarr):
        """Truncate back to the N grid (2/3 rule)."""
        N = self.N
        B = specarr.shape[0]
        trunc = specarr.new_zeros((B, 2, N, N // 2 + 1))
        trunc[:, :, 0:N // 2, 0:N // 2] = specarr[:, :, 0:N // 2, 0:N // 2]
        trunc[:, :, -N // 2:, 0:N // 2] = specarr[:, :, -N // 2:, 0:N // 2]
        return trunc

    def xyderiv(self, specarr):
        if not self.dealias:
            xderiv = torch.fft.irfft2(self.ik * specarr, s=(self.N, self.N))
            yderiv = torch.fft.irfft2(self.il * specarr, s=(self.N, self.N))
        else:
            pad = self.specpad(specarr)
            n_pad = 3 * self.N // 2
            xderiv = torch.fft.irfft2(self.ik_pad * pad, s=(n_pad, n_pad))
            yderiv = torch.fft.irfft2(self.il_pad * pad, s=(n_pad, n_pad))
        return xderiv, yderiv

    # dynamics
    def gettend(self, pvspec=None):
        """Spectral PV tendency on z=0,H."""
        if pvspec is None:
            pvspec = self.pvspec
        psispec = self.invert(pvspec)
        psix, psiy = self.xyderiv(psispec)
        pvx, pvy = self.xyderiv(pvspec)
        jacobian = psix * pvy - psiy * pvx
        jacobianspec = torch.fft.rfft2(jacobian)
        if self.dealias:
            jacobianspec = self.spectrunc(jacobianspec)
        dpvspecdt = (1.0 / self.tdiab) * (self.pvspec_eq - pvspec) - jacobianspec
        if self.ekman:
            # Ekman damping at the boundaries (out of place, for autograd).
            damp0 = self.r * self.ksqlsq * psispec[:, 0]
            if self.symmetric:
                damp1 = -self.r * self.ksqlsq * psispec[:, 1]
            else:
                # asymmetric jet (U=0 at sfc): no Ekman layer at the lid
                damp1 = torch.zeros_like(damp0)
            dpvspecdt = dpvspecdt + torch.stack((damp0, damp1), dim=1)
        self.u = -psiy
        self.v = psix
        return dpvspecdt

    def timestep(self, pvspec=None):
        """One RK4 step with integrating-factor hyperdiffusion."""
        if pvspec is None:
            pvspec = self.pvspec
        k1 = self.dt * self.gettend(pvspec)
        k2 = self.dt * self.gettend(pvspec + 0.5 * k1)
        k3 = self.dt * self.gettend(pvspec + 0.5 * k2)
        k4 = self.dt * self.gettend(pvspec + k3)
        pvspecnew = pvspec + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return self.hyperdiff * pvspecnew

    def advance(self, pv=None, timesteps=None):
        """Advance `timesteps` steps and return PV on the grid.

        Args:
            pv: (n_ens, 2, N, N) or (2, N, N) PV on the grid. If None the
                internal `pvspec` is advanced in place (matching `sqg.SQG`).
            timesteps: override `self.timesteps`.

        Returns:
            PV on the grid, same shape/rank as the input.
        """
        nsteps = self.timesteps if timesteps is None else timesteps
        squeeze = False
        if pv is None:
            pvspec = self.pvspec
            inplace = True
        else:
            inplace = False
            pv = torch.as_tensor(pv)
            if pv.dim() == 3:
                pv = pv.unsqueeze(0)
                squeeze = True
            pv = pv.to(dtype=self.dtype, device=self.device)
            pvspec = torch.fft.rfft2(pv).to(self.cdtype)

        for _ in range(nsteps):
            pvspec = self.timestep(pvspec)

        if inplace:
            self.pvspec = pvspec
        self.t = self.t + float(self.dt) * nsteps

        out = torch.fft.irfft2(pvspec, s=(self.N, self.N))
        return out.squeeze(0) if squeeze else out
