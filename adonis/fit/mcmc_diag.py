"""MCMC diagnostics: rank-normalised split-Rhat, bulk/tail ESS, autocorrelation time.

Implemented here rather than imported because this environment has no arviz / numpyro / blackjax.  Every
function is checked in tests/test_mcmc_diag.py against cases with a KNOWN answer -- iid draws (ESS = N),
AR(1) with correlation rho (ESS = N (1-rho)/(1+rho)), and chains deliberately offset from each other
(Rhat >> 1).  A diagnostic that is silently wrong would not announce itself: it would just hand back a
plausible efficiency number, which is the entire quantity a sampler comparison is about.

Definitions follow Vehtari, Gelman, Simpson, Carpenter & Burkner (2021), "Rank-normalization, folding,
and localization: an improved Rhat for assessing convergence of MCMC":

  split-Rhat   each chain is split in half before the between/within comparison, so a chain that drifts
               within itself is caught even when the chains agree with each other overall.
  rank-norm    draws are replaced by their normal scores across all chains, which makes the statistic
               invariant to monotone reparameterisation and finite even for heavy tails.
  bulk-ESS     ESS of the rank-normalised draws -- efficiency for the centre of the distribution.
  tail-ESS     min of the ESS of the 5% and 95% quantile indicators -- efficiency where interval
               endpoints come from.  Bulk alone hides a sampler that mixes well in the core and badly in
               the tails, which is exactly the failure mode of a random walk on a correlated posterior.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def _as_chains(x):
    """(nchain, ndraw) float array."""
    a = np.asarray(x, float)
    if a.ndim == 1:
        a = a[None, :]
    return a


def _split(a):
    """Split each chain in half: (nchain, ndraw) -> (2 nchain, ndraw//2)."""
    n = a.shape[1] // 2
    return np.concatenate([a[:, :n], a[:, n:2 * n]], axis=0)


def _rank_normalise(a):
    """Normal scores of the pooled draws, reshaped back to (nchain, ndraw).

    Average ranks for ties, then the Blom transform r -> Phi^-1((r - 3/8)/(N + 1/4)).
    """
    flat = a.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, float)
    ranks[order] = np.arange(1, flat.size + 1, dtype=float)
    # average ranks over ties, so a sampler that repeats a value (a REJECTED Metropolis step) is not
    # given a spurious ordering -- rejections are common and would otherwise bias the statistic
    u, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    if cnt.max() > 1:
        sums = np.zeros(u.size)
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    z = ndtri((ranks - 3.0 / 8.0) / (flat.size + 0.25))
    return z.reshape(a.shape)


def _autocov(y):
    """Autocovariance of one chain at all lags, by FFT."""
    n = y.size
    y = y - y.mean()
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, m)
    ac = np.fft.irfft(f * np.conjugate(f), m)[:n]
    return ac / n


def _ess_from_chains(a):
    """ESS by the Geyer initial-monotone-positive-sequence estimator (Stan's rule).

    The variance estimate is the same rank/split-aware one Rhat uses, so ESS and Rhat cannot disagree
    about what the chains are.
    """
    m, n = a.shape
    if n < 4:
        return float(m * n)
    acov = np.stack([_autocov(a[i]) for i in range(m)])            # (m, n)
    chain_var = acov[:, 0] * n / (n - 1.0)
    var_plus = chain_var.mean()
    if m > 1:
        var_plus = var_plus * (n - 1.0) / n + a.mean(axis=1).var(ddof=1)
    if not np.isfinite(var_plus) or var_plus <= 0:
        return float(m * n)
    # rho_t averaged over chains
    rho = 1.0 - (chain_var[:, None] - acov).mean(axis=0) / var_plus     # rho[0] == 1
    # Geyer: sum consecutive PAIRS while positive, then enforce monotonicity.
    # THE PAIRING STARTS AT LAG 0: P_t = rho_{2t} + rho_{2t+1}, so P_0 = 1 + rho_1 and
    #     tau = -1 + 2 sum_t P_t = 1 + 2 sum_{t>=1} rho_t,
    # which is the definition.  Starting at lag 1 instead drops that leading 1 and returns
    # tau = -1 + 2 sum_{t>=1} rho_t -- measured tau = 0.91 against a true 3.00 on AR(1) rho=0.5, i.e. ESS
    # overestimated 3.3x, and the error VANISHES as rho -> 1 (1.02x at rho=0.95), so it would have looked
    # harmless on exactly the badly-mixing chains a sampler comparison cares least about and wrong on the
    # well-mixing ones it cares most about.
    t = 0
    pair = []
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0 and t > 0:
            break
        pair.append(p)
        t += 2
    if not pair:
        return float(m * n)
    pair = np.array(pair)
    pair = np.minimum.accumulate(pair)          # monotone non-increasing
    tau = -1.0 + 2.0 * pair.sum()
    tau = max(tau, 1.0 / np.log10(max(m * n, 11)))
    return float(m * n / tau)


def split_rhat(x):
    """Rank-normalised split-Rhat.  1.0 for converged chains; >1.01 is the usual action threshold."""
    a = _rank_normalise(_split(_as_chains(x)))
    m, n = a.shape
    if n < 2:
        return np.nan
    W = a.var(axis=1, ddof=1).mean()
    B = n * a.mean(axis=1).var(ddof=1) if m > 1 else 0.0
    if W <= 0:
        return np.nan
    return float(np.sqrt(((n - 1.0) / n * W + B / n) / W))


def ess_bulk(x):
    """ESS of the rank-normalised split chains -- efficiency in the body of the distribution."""
    return _ess_from_chains(_rank_normalise(_split(_as_chains(x))))


def ess_tail(x):
    """min ESS of the 5% and 95% quantile indicators -- efficiency where interval ENDS come from.

    An indicator series is 0/1, so its ESS measures how well the sampler crosses that quantile.  Bulk ESS
    can look healthy while the tails are barely explored, and interval endpoints are what a physics result
    quotes, so this is the number that should gate a credible interval.
    """
    a = _as_chains(x)
    q05, q95 = np.quantile(a, 0.05), np.quantile(a, 0.95)
    out = []
    for q in (q05, q95):
        ind = (a <= q).astype(float)
        if ind.std() == 0:
            continue
        out.append(_ess_from_chains(_split(ind)))
    return float(min(out)) if out else np.nan


def ess_basic(x):
    """ESS of the RAW draws (no rank normalisation) -- what tau_int is defined against."""
    return _ess_from_chains(_split(_as_chains(x)))


def tau_int(x):
    """Integrated autocorrelation time in draws: N_total / ESS_basic."""
    a = _as_chains(x)
    e = ess_basic(a)
    return float(a.size / e) if e > 0 else np.nan


def mcse_mean(x):
    """Monte-Carlo standard error of the mean, from bulk ESS.  Two samplers agree on a marginal only if
    their means differ by a few of THESE, not by a few posterior sigmas."""
    a = _as_chains(x)
    e = ess_bulk(a)
    return float(a.std(ddof=1) / np.sqrt(e)) if e > 0 else np.nan


def summarise(draws, names):
    """Per-parameter diagnostics for (nchain, ndraw, npar) draws."""
    d = np.asarray(draws, float)
    out = []
    for k, nm in enumerate(names):
        a = d[:, :, k]
        out.append(dict(name=nm, mean=float(a.mean()), sd=float(a.std(ddof=1)),
                        rhat=split_rhat(a), ess_bulk=ess_bulk(a), ess_tail=ess_tail(a),
                        tau=tau_int(a), mcse=mcse_mean(a)))
    return out
