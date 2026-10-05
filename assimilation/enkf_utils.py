import numpy as np
from scipy.linalg import lapack, inv

# function definitions.


def cartdist(x1, y1, x2, y2, xmax, ymax, periodic):
    """cartesian distance on doubly periodic plane"""
    dx = np.abs(x1 - x2)
    dy = np.abs(y1 - y2)

    if periodic[0]:
        dx = np.where(dx > 0.5 * xmax, xmax - dx, dx)
    if periodic[1]:
        dy = np.where(dy > 0.5 * ymax, ymax - dy, dy)
    
    return np.sqrt(dx ** 2 + dy ** 2)


def gaspcohn(r):
    """
    Gaspari-Cohn taper function.
    very close to exp(-(r/c)**2), where c = sqrt(0.15)
    r should be >0 and normalized so taper = 0 at r = 1
    """
    rr = 2.0 * r
    rr += 1.0e-13  # avoid divide by zero warnings from numpy
    taper = np.where(
        r <= 0.5,
        (((-0.25 * rr + 0.5) * rr + 0.625) * rr - 5.0 / 3.0) * rr ** 2 + 1.0,
        np.zeros(r.shape, r.dtype),
    )
    taper = np.where(
        np.logical_and(r > 0.5, r < 1.0),
        ((((rr / 12.0 - 0.5) * rr + 0.625) * rr + 5.0 / 3.0) * rr - 5.0) * rr
        + 4.0
        - 2.0 / (3.0 * rr),
        taper,
    )
    return taper


def enkf_update(
    xens, hxens, obs, oberrs, covlocal, vcovlocal_fact, obcovlocal=None,
    return_wts=False,
):
    """serial potter method or LETKF (if obcovlocal is None)

    If return_wts is True (LETKF branch only) also return the local analysis
    weight matrices, shape (2, ndim1, nanals, nanals). Applying wts[k, n] to the
    prior perturbations of an *earlier* time gives the smoothed ensemble at that
    time -- the "no-cost smoother" of Kalnay and Yang (2010), doi:10.1002/qj.652.
    """
    nanals, nlevs, ndim = xens.shape
    nobs = obs.shape[-1]
    xmean = xens.mean(axis=0)
    xprime = xens - xmean
    hxmean = hxens.mean(axis=0)
    hxprime = hxens - hxmean
    # One entry per level. `nlevs` is already unpacked above; the literal
    # 2s below were the SQG level count. For nlevs == 2 every expression here
    # is unchanged, so SQG results are bit-identical.
    fact = np.ones(nlevs, float)
    if nlevs > 2:
        raise NotImplementedError(
            f"vertical localization is defined for 1 or 2 levels, got {nlevs}: "
            "`fact[1 - kob]` means 'the other level' and does not generalize"
        )

    if return_wts and obcovlocal is not None:
        raise ValueError("return_wts is only supported for the LETKF update "
                         "(obcovlocal must be None)")

    if obcovlocal is not None:  # serial EnSRF update
        # print('serial EnSRF update')
        for kob in range(nlevs):
            fact[:] = 1.0
            if nlevs == 2:
                fact[1 - kob] = vcovlocal_fact
            for nob, ob, oberr in zip(np.arange(nobs), obs[kob], oberrs):
                ominusf = ob - hxmean[kob, nob].copy()
                hxens = hxprime[:, kob, nob].copy().reshape((nanals, 1))
                hpbht = (hxens ** 2).sum() / (nanals - 1)
                gainfact = (
                    (hpbht + oberr)
                    / hpbht
                    * (1.0 - np.sqrt(oberr / (hpbht + oberr)))
                )
                # state space update
                # only update points closer than localization radius to ob
                mask = covlocal[nob, :] > 1.0e-10
                for k in range(nlevs):
                    pbht = (np.arctan(xprime[:, k, mask]).T * hxens[:, 0]).sum(axis=1) / float(   # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                        nanals - 1
                    )
                    kfgain = fact[k] * covlocal[nob, mask] * \
                        pbht / (hpbht + oberr)
                    # print(kfgain)
                    xmean[k, mask] = xmean[k, mask] + kfgain * ominusf
                    # print(xmean[k, mask])
                    xprime[:, k, mask] = xprime[:, k, mask] - \
                        gainfact * kfgain * hxens
                # observation space update
                # only update obs within localization radius
                mask = obcovlocal[nob, :] > 1.0e-10
                for k in range(nlevs):
                    pbht = (hxprime[:, k, mask].T * hxens[:, 0]).sum(axis=1) / float(
                        nanals - 1
                    )
                    kfgain = fact[k] * obcovlocal[nob, mask] * \
                        pbht / (hpbht + oberr)
                    hxmean[k, mask] = hxmean[k, mask] + kfgain * ominusf
                    hxprime[:, k, mask] = (
                        hxprime[:, k, mask] - gainfact * kfgain * hxens
                    )
        return xmean + xprime

    else:  # LETKF update
        # print('LETKF update start')

        ndim1 = covlocal.shape[-1]
        hx = np.empty((nanals, nlevs * nobs), float)
        omf = np.empty(nlevs * nobs, float)
        oberrvar = np.empty(nlevs * nobs, float)
        covlocal_tmp = np.empty((nlevs * nobs, nlevs, ndim1), float)
        for kob in range(nlevs):
            fact[:] = 1.0
            if nlevs == 2:
                fact[1 - kob] = vcovlocal_fact
            oberrvar[kob * nobs: (kob + 1) * nobs] = oberrs[:]
            omf[kob * nobs: (kob + 1) * nobs] = obs[kob, :] - hxmean[kob, :]
            hx[:, kob * nobs: (kob + 1) * nobs] = hxprime[:, kob, :]
            for k in range(nlevs):
                covlocal_tmp[kob * nobs: (kob + 1) * nobs, k, :] = (
                    fact[k] * covlocal[:, :]
                )

        def calcwts(hx, Rinv, ominusf):
            YbRinv = np.dot(hx, Rinv)
            pa = (nanals - 1) * np.eye(nanals) + np.dot(YbRinv, hx.T)
            # print('Rinv')
            # print(Rinv)
            # print('hx')
            # print(hx)
            # print(np.linalg.cond(pa))
            evals, eigs, info = lapack.dsyevd(pa)
            # evals, eigs, info, isuppz, info = lapack.dsyevr(pa)
            evals = evals.clip(min=np.finfo(evals.dtype).eps)
            painv = np.dot(np.dot(eigs, np.diag(np.sqrt(1.0 / evals))), eigs.T)
            tmp = np.dot(np.dot(np.dot(painv, painv.T), YbRinv), ominusf)
            return np.sqrt(nanals - 1) * painv + tmp[:, np.newaxis]
        # print(ndim1)
        wts_all = np.empty((nlevs, ndim1, nanals, nanals), float) if return_wts else None
        for n in range(ndim1):
            for k in range(nlevs):
                mask = covlocal_tmp[:, k, n] > 1.0e-10
                # print(mask)
                Rinv = np.diag(covlocal_tmp[mask, k, n] / oberrvar[mask])
                ominusf = omf[mask]
                wts = calcwts(hx[:, mask], Rinv, ominusf)
                # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                xens[:, k, n] = xmean[k, n] + np.dot(wts.T, xprime[:, k, n])
                # print(np.dot(wts.T, xprime[:, k, n]))
                if return_wts:
                    wts_all[k, n] = wts

        if return_wts:
            return xens, wts_all
        return xens


def bulk_ensrf(
    xens, indxobi, obs, oberrs, covlocal1, vcovlocal_fact, pv_scalefact
):
    """bulk potter method (global matrix solution)

    NOTE: still assumes exactly 2 levels (the `2 * nobs1` / `2 * ndim1`
    flattening below). Only reached with LETKF.global_enkf = True, which is
    False by default; enkf_update above is the generalized path.
    """
    # print('bulk potter method')
    nanals, nlevs, ndim1 = xens.shape
    if nlevs != 2:
        raise NotImplementedError(
            f"bulk_ensrf assumes 2 levels, got {nlevs}; use the LETKF path"
        )
    nobs1 = obs.shape[-1]
    nobs = 2 * nobs1
    ndim = 2 * ndim1
    xmean2 = xens.mean(axis=0)
    xprime2 = xens - xmean2

    # create H operator
    iob = np.zeros(ndim1, bool)
    iob[indxobi] = True
    indxob = np.concatenate((iob, iob))

    # create cov localization matrix
    covlocal = np.block(
        [
            [covlocal1, vcovlocal_fact * covlocal1],
            [vcovlocal_fact * covlocal1, covlocal1],
        ]
    )

    # create 1d state vector arrays
    xmean = xmean2.reshape(ndim)
    xprime = xprime2.reshape((nanals, ndim))
    obs = obs.reshape(nobs)
    oberrstd = np.sqrt(np.concatenate((oberrs, oberrs)))
    # normalize obs by ob erro stdev
    obs = obs / oberrstd

    # forward operator
    hxmean = pv_scalefact * xmean[indxob]
    hxprime = pv_scalefact * xprime[:, indxob] / oberrstd

    eye = np.eye(nobs)
    Pb = covlocal * np.dot(xprime.T, xprime) / (nanals - 1)
    D = pv_scalefact ** 2 * Pb[np.ix_(indxob, indxob)] + eye
    PbHT = pv_scalefact * Pb[:, indxob]

    # see https://doi.org/10.1175/JTECH-D-16-0140.1 eqn 5

    # using Cholesky and LU decomp
    Dsqrt, info = lapack.dpotrf(D, overwrite_a=0)
    Dinv, info = lapack.dpotri(Dsqrt)
    # lapack only returns the upper triangular part
    Dinv += np.triu(Dinv, k=1).T
    kfgain = np.dot(PbHT, Dinv)
    Dsqrt = np.triu(Dsqrt)
    DplusDsqrtinv = inv(D+Dsqrt)  # uses lapack dgetrf,dgetri
    reducedgain = np.dot(PbHT, DplusDsqrtinv)

    # Using eigenanalysis
    # evals, eigs, info = lapack.dsyevd(D)
    # evals, eigs, info, isuppz, info = lapack.dsyevr(D)
    # evals = evals.clip(min=np.finfo(evals.dtype).eps)
    # Dinv = (eigs * (1.0 / evals)).dot(eigs.T)
    # kfgain = np.dot(PbHT, Dinv)
    # DplusDsqrtinv = (eigs * (1.0 / (evals + np.sqrt(evals)))).dot(eigs.T)
    # reducedgain = np.dot(PbHT, DplusDsqrtinv)

    # mean and perturbation update
    xmean += np.dot(kfgain, obs - hxmean)
    xprime -= np.dot(reducedgain, hxprime.T).T

    # back to 2d state vectors
    xmean2 = xmean.reshape((2, ndim1))
    xprime2 = xprime.reshape((nanals, 2, ndim1))
    xens = xmean2 + xprime2

    return xens
