"""PHYSICAL FIT (logbook 17) — steps 2/3: fit engines M0/M1/M2 + Gate II + flagging.

Fake-data modes (PHYSFIT_MODE):
  closure  : data = exact-reweight model at INJECTED knobs (PHYSFIT_INJECT="M_A_qe=1.2,res_norm=0.8")
  inject2x : data = nominal Asimov with bins x2 in one observable above a threshold
             (PHYSFIT_2X="dpt>300", native units) — the known "unknown unknown"
  Errors: sigma^2 = (SYST*data)^2 + ADoNIS-MC^2, recomputed on the (injected) data.

Methods (PHYSFIT_METHOD, comma list or "all"):
  M0 : traditional — LM minimize chi2_data + prior penalty on the Gate-I knob subset (control).
  M1 : physical   — M0, then Gate II: per-knob Cochran's Q on per-bin demands + vector split-fit
       Q_split (one-step GN on disjoint low/high-half regions); freeze failers, refit, iterate.
       Then FLAG contiguous |r/sigma|>2 regions (unknown-unknown candidates).
  M2 : robust     — Huber IRLS (c=1.345) in the loss (the principled in-loss competitor).

Gate-I subset: knobs with shrinkage < 0.5 from GATE1_NPZ (blind to what was injected).
All fits include the prior term chi2_prior = sum ((theta-nom)/prior)^2 for FITTED knobs;
frozen knobs stay at nominal. LIVE per-iteration progress.

Calibration caveat (documented): Q_split one-step estimates share the prior anchor -> conservative
(under-flags); acceptable for v1, checked empirically in the closure step.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy import stats as sstats
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from analysis.t2k.differentiability import bank_plot as BP, bank_reweight as BR
from analysis.t2k.differentiability.full_knobs import nominal_knobs
from analysis.t2k.differentiability import info_content as IC
sys.path.insert(0, str(Path(__file__).resolve().parent))
from physical_fit import (SPEC, NPAR, PNAMES, PRIOR, theta_nominal, knobs_of,
                          build_physfit_datasets, N_BINS, SYST)

BANKDIR = os.environ.get("ADONIS_EVENT_BANK", "output/event_bank")
GATE1_NPZ = os.environ.get("GATE1_NPZ", "output/altgen/physfit_gate1.npz")
MODE = os.environ.get("PHYSFIT_MODE", "closure")
METHODS = os.environ.get("PHYSFIT_METHOD", "all")
INJECT = os.environ.get("PHYSFIT_INJECT", "M_A_qe=1.2,res_norm=0.8")
INJ2X = os.environ.get("PHYSFIT_2X", "dpt>300")
LABEL = os.environ.get("ADONIS_LABEL", f"physfit_{MODE}")
NIT = int(os.environ.get("ALTGEN_NIT", "12"))
F_RESP = 0.3            # responsive-bin threshold: |J_bk|*prior_k > F_RESP*sigma_b
P_GATE = 0.01           # Gate II p-value threshold (Q_k and Q_split)
HUBER_C = 1.345

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def parse_inject(s, nom):
    th = theta_nominal(nom)
    inj = {}
    for tok in s.split(","):
        k, v = tok.split("="); inj[k.strip()] = float(v)
    for j, (key, idx, *_ ) in enumerate(SPEC):
        nm = f"{key}[{idx}]" if idx is not None else key
        if nm in inj:
            th[j] = inj[nm]
    return th, inj


def refresh_sigma(ds):
    """Recompute sigma/Cinv from the CURRENT d['data'] (syst part) + fixed MC part (+ GENIE stat)."""
    for d in ds:
        var = (SYST * d["data"])**2 + d["mcerr"]**2
        if "stat_g" in d:
            var = var + d["stat_g"]**2
        d["sigma"] = np.sqrt(var); d["Cinv"] = np.diag(1.0 / np.maximum(var, 1e-300))


class Engine:
    """Shared differentiable model/Jacobian over the SPEC vector, restricted to a fit subset."""
    def __init__(self, ds, JB, grids, nom):
        self.ds, self.JB, self.grids, self.nom = ds, JB, grids, nom
        self.th0 = theta_nominal(nom)
        self.row0 = np.cumsum([0] + [d["nbin"] for d in ds])
        wf = lambda th, JB: BR.bank_weight(JB, knobs_of(th, nom), grids)
        self.wf_jit = jax.jit(wf)
        self.jvp = jax.jit(lambda th, tang, JB: jax.jvp(lambda t: wf(t, JB), (th,), (tang,))[1])

    def model(self, th):
        w = np.asarray(self.wf_jit(jnp.asarray(th), self.JB))
        return np.concatenate([IC.bin_w(d, w) for d in self.ds])

    def jac(self, th, subset):
        J = np.zeros((self.row0[-1], len(subset)))
        for c, k in enumerate(subset):
            g = np.asarray(self.jvp(jnp.asarray(th), jnp.zeros(NPAR).at[k].set(1.0), self.JB))
            for j, d in enumerate(self.ds):
                J[self.row0[j]:self.row0[j+1], c] = IC.bin_w0(d, g)
        return J

    def data_sigma(self):
        return (np.concatenate([d["data"] for d in self.ds]),
                np.concatenate([d["sigma"] for d in self.ds]))


def lm_fit(eng, subset, tag, huber=False, nit=NIT, mask=None):
    """LM on chi2_data(+Huber) + prior penalty over `subset` knobs, restricted to `mask` bins.
    Returns th(full), V, J(full rows), m(full), chi2s (masked)."""
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    prior_w = 1.0 / PRIOR[subset]**2
    th = eng.th0.copy(); lam = 1e-3
    def chi2_terms(thv):
        m = eng.model(thv); u = (m - data) / sigma
        hw = np.minimum(1.0, HUBER_C / np.maximum(np.abs(u), 1e-12)) if huber else np.ones_like(u)
        c_data = float(np.sum((hw * u**2)[mask]))
        c_pri = float(np.sum(prior_w * (thv[subset] - eng.th0[subset])**2))
        return c_data + c_pri, c_data, m, hw
    c_cur, c_data, m, hw = chi2_terms(th)
    log(f"  [{tag}] start chi2 {c_cur:.1f} (data {c_data:.1f}, {int(mask.sum())} bins)")
    for it in range(nit):
        c_before = c_cur
        J = eng.jac(th, subset)
        W = np.where(mask, hw / sigma**2, 0.0)
        A = J.T @ (J * W[:, None]) + np.diag(prior_w)
        g = J.T @ (W * (m - data)) + prior_w * (th[subset] - eng.th0[subset])
        for _ in range(12):
            dth = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            th_try = th.copy(); th_try[subset] = th[subset] + dth
            c_try, cd_try, m_try, hw_try = chi2_terms(th_try)
            if c_try < c_cur:
                th, c_cur, c_data, m, hw = th_try, c_try, cd_try, m_try, hw_try
                lam = max(lam / 3, 1e-8); break
            lam *= 5
        log(f"  [{tag}] it {it:2d} chi2={c_cur:9.2f} (data {c_data:9.2f}) " +
            " ".join(f"{PNAMES[k]}={th[k]:.3f}" for k in subset))
        if np.linalg.norm(dth) < 1e-6 or (c_before - c_cur) / max(c_before, 1e-9) < 1e-3:
            log(f"  [{tag}] converged/plateau at it {it}"); break
    J = eng.jac(th, subset)
    W = np.where(mask, hw / sigma**2, 0.0)
    A = J.T @ (J * W[:, None]) + np.diag(prior_w)
    V = np.linalg.pinv(A, rcond=1e-12)
    return th, V, J, m, c_cur, c_data


def gate2_Q(eng, th, subset, J, m, mask=None):
    """Per-knob Cochran's Q on per-bin demands at the BFP. Returns dict knob->(Q,ndf,p,I2,nresp)."""
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    r = data - m
    out = {}
    for c, k in enumerate(subset):
        Jk = J[:, c]
        resp = (np.abs(Jk) * PRIOR[k] > F_RESP * sigma) & mask
        n = int(resp.sum())
        if n < 3:
            out[k] = dict(Q=0.0, ndf=0, p=1.0, I2=0.0, nresp=n); continue
        dth = r[resp] / Jk[resp]
        w = (Jk[resp] / sigma[resp])**2
        dhat = np.sum(w * dth) / np.sum(w)
        Q = float(np.sum(w * (dth - dhat)**2)); ndf = n - 1
        p = float(sstats.chi2.sf(Q, ndf))
        I2 = max(0.0, (Q - ndf) / max(Q, 1e-12))
        out[k] = dict(Q=Q, ndf=ndf, p=p, I2=I2, nresp=n, dhat=float(dhat))
    return out


def gate2_split(eng, th, subset, J, m, mask=None):
    """Vector split-fit: one-step GN estimates on low/high-half regions (per observable), prior-anchored."""
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    r = data - m
    lo_mask = np.zeros(len(data), bool)
    for j, d in enumerate(eng.ds):
        s, e = eng.row0[j], eng.row0[j + 1]
        lo_mask[s:s + d["nbin"] // 2] = True
    prior_w = np.diag(1.0 / PRIOR[subset]**2)
    est = {}
    for name, msk in (("lo", lo_mask & mask), ("hi", (~lo_mask) & mask)):
        Ji = J[msk]; Ci = 1.0 / sigma[msk]**2
        Ai = Ji.T @ (Ji * Ci[:, None]) + prior_w
        Vi = np.linalg.pinv(Ai, rcond=1e-12)
        dthi = Vi @ (Ji.T @ (Ci * r[msk]))
        est[name] = (th[subset] + dthi, Vi)
    dtheta = est["lo"][0] - est["hi"][0]
    Vsum = est["lo"][1] + est["hi"][1]
    Q = float(dtheta @ np.linalg.pinv(Vsum, rcond=1e-12) @ dtheta)
    ndf = len(subset)
    p = float(sstats.chi2.sf(Q, ndf))
    evals, evecs = np.linalg.eigh(np.linalg.pinv(Vsum, rcond=1e-12))
    worst = evecs[:, -1]
    zk = dtheta / np.sqrt(np.maximum(np.diag(Vsum), 1e-300))
    return dict(Q=Q, ndf=ndf, p=p, dtheta=dtheta, zk=zk, worst=worst,
                th_lo=est["lo"][0], th_hi=est["hi"][0])


def flags(eng, m, mask=None):
    """Contiguous runs of >=2 bins with |r/sigma|>2 at the BFP (excised bins never re-flagged)."""
    data, sigma = eng.data_sigma()
    if mask is None:
        mask = np.ones(len(data), bool)
    pull = (data - m) / sigma
    out = []
    for j, d in enumerate(eng.ds):
        s = eng.row0[j]; p = pull[s:s + d["nbin"]]
        bad = (np.abs(p) > 2.0) & mask[s:s + d["nbin"]]
        i = 0
        while i < len(bad):
            if bad[i]:
                k = i
                while k + 1 < len(bad) and bad[k + 1]:
                    k += 1
                if k - i + 1 >= 2:
                    out.append(dict(obs=d["name"], j=j, i0=i, i1=k,
                                    lo=float(d["edges"][i]), hi=float(d["edges"][k + 1]),
                                    nbins=k - i + 1, mean_pull=float(np.mean(p[i:k + 1]))))
                i = k + 1
            else:
                i += 1
    return out


def main():
    B = BP.load_bank(BANKDIR); JB = BR.to_jax(B); grids = BR.default_grids(); nom = nominal_knobs()
    w0 = np.asarray(BR.weight_jit(JB, nom, grids))
    log(f"bank {len(w0)} events | mode={MODE} methods={METHODS} label={LABEL}")
    ds = build_physfit_datasets(B, w0, log)
    eng = Engine(ds, JB, grids, nom)

    # ---- Gate-I subset (blind to injection) ------------------------------------------------------- #
    g1 = np.load(GATE1_NPZ, allow_pickle=True)
    subset = [int(i) for i in np.where(g1["shrink"] < 0.5)[0]]
    log(f"Gate-I subset ({len(subset)}): " + " ".join(PNAMES[k] for k in subset))

    # ---- fake data -------------------------------------------------------------------------------- #
    truth = eng.th0.copy(); inj_desc = "asimov"
    if MODE == "closure":
        truth, inj = parse_inject(INJECT, nom)
        w_inj = np.asarray(BR.weight_jit(JB, knobs_of(truth, nom), grids))
        for j, d in enumerate(ds):
            d["data"] = IC.bin_w(d, w_inj)
        inj_desc = INJECT
    elif MODE == "inject2x":
        obs, thr = INJ2X.split(">"); thr = float(thr)
        for d in ds:
            if d["key"] == obs.strip():
                hit = d["edges"][:-1] >= thr
                d["data"] = d["data"] * np.where(hit, 2.0, 1.0)
                log(f"  x2 injection: {d['name']} bins with edge>={thr} ({int(hit.sum())} bins)")
        inj_desc = INJ2X
    elif MODE == "genie":
        # GENIE 3M as data on the physfit binning: same extraction as build_fakedata (single source
        # of truth), identical overflow clipping; data sigma additionally carries GENIE Poisson stat.
        from build_fakedata import extract_cc0pi, extract_cc1pi
        gst = os.environ.get("ADONIS_GST", "output/altgen/genie_t2k_12C_ar23_CCQERES_3M.gst.root")
        E = extract_cc0pi(gst); E1 = extract_cc1pi(E)
        gvals = {"dpt": E["dpt"][E["sel"]], "dat": E["dat"][E["sel"]],
                 "pn": E1["vals1"]["pn"], "dptt": E1["vals1"]["dptt"], "daT": E1["vals1"]["daT"]}
        for d in ds:
            edges = d["edges"]; eps = (edges[-1] - edges[0]) * 1e-12
            vc = np.clip(gvals[d["key"]], edges[0] + eps, edges[-1] - eps)
            cnt, _ = np.histogram(vc, bins=edges)
            if d["key"] in ("dpt", "dat"):
                bw_unit = np.diff(edges) / (1000.0 if d["key"] == "dpt" else 1.0)
                scale = E["per_event"] / bw_unit
                d["data"] = cnt * scale                            # 1e-38/unit/nucleon
            else:
                scale = E1["per_event_nb_CH"] / np.diff(edges)
                d["data"] = cnt * scale + d["offset"]              # GENIE-C + frozen free-H (nb/CH)
            d["stat_g"] = np.sqrt(cnt) * scale
        inj_desc = f"genie:{Path(gst).name}"
    refresh_sigma(ds)
    log(f"fake data ready: {inj_desc}")

    # ---- run methods ------------------------------------------------------------------------------ #
    want = ["M0", "M1", "M2"] if METHODS == "all" else METHODS.split(",")
    results = {}
    for meth in want:
        log(f"==== {meth} ====")
        if meth in ("M0", "M2"):
            th, V, J, m, c, cd = lm_fit(eng, subset, meth, huber=(meth == "M2"))
            results[meth] = dict(th=th, V=V, sub=subset, chi2=c, chi2_data=cd,
                                 Qk=gate2_Q(eng, th, subset, J, m),
                                 split=gate2_split(eng, th, subset, J, m), flags=flags(eng, m))
        else:  # M1: gated fit with EXCISE-REFIT — freeze incoherent knobs; flag+excise bad regions;
               # refit the clean bins with the FULL Gate-I subset (knobs may become coherent once the
               # mismodeled region is removed) until no new flags.
            mask = np.ones(eng.row0[-1], bool)
            excised = []
            fl = []
            for xr in range(3):
                sub = list(subset); frozen = []
                for rnd in range(4):
                    if not sub:
                        # all frozen -> evaluate at NOMINAL (not the discarded biased point)
                        th = eng.th0.copy(); m = eng.model(th)
                        V = np.zeros((0, 0)); J = np.zeros((eng.row0[-1], 0))
                        dd, ss = eng.data_sigma()
                        cd = float(np.sum((((m - dd) / ss)**2)[mask])); c = cd
                        Qk = {}; sp = dict(Q=float("nan"), ndf=0, p=float("nan"), zk=np.array([]))
                        log(f"  [M1x{xr}] all knobs frozen -> evaluated at nominal")
                        break
                    th, V, J, m, c, cd = lm_fit(eng, sub, f"M1x{xr}r{rnd}", mask=mask)
                    Qk = gate2_Q(eng, th, sub, J, m, mask=mask)
                    sp = gate2_split(eng, th, sub, J, m, mask=mask)
                    fail = [k for k in sub if Qk[k]["p"] < P_GATE]
                    # split failure: freeze the knob with the largest |z| if the vector splits
                    if sp["p"] < P_GATE:
                        kz = sub[int(np.argmax(np.abs(sp["zk"])))]
                        if kz not in fail:
                            fail.append(kz)
                    log(f"  [M1x{xr}r{rnd}] Qk fails: {[PNAMES[k] for k in fail]}  "
                        f"Q_split={sp['Q']:.1f}/{sp['ndf']} (p={sp['p']:.3g})")
                    if not fail:
                        break
                    for k in fail:
                        sub.remove(k); frozen.append(k)
                fl = flags(eng, m, mask=mask)
                if not fl:
                    log(f"  [M1x{xr}] no flags on the clean region -> converged"); break
                for f in fl:
                    excised.append(f)
                    mask[eng.row0[f["j"]] + f["i0"]: eng.row0[f["j"]] + f["i1"] + 1] = False
                log(f"  [M1x{xr}] excised {sum(f['nbins'] for f in fl)} bins "
                    f"({int((~mask).sum())} total) -> refit clean region")
            results[meth] = dict(th=th, V=V, sub=sub, frozen=frozen, chi2=c, chi2_data=cd,
                                 Qk=Qk, split=sp, flags=fl, excised=excised)

    # ---- report ----------------------------------------------------------------------------------- #
    print(f"\n==== PHYSICAL-FIT run [{LABEL}] mode={MODE} ({inj_desc}) ====")
    for meth, R in results.items():
        print(f"\n-- {meth} --  chi2_data={R['chi2_data']:.1f}")
        print(f"{'knob':>16} {'truth':>7} {'BFP':>8} {'+/-':>7} {'bias/sig':>8}   Qk_p")
        for c, k in enumerate(R["sub"]):
            s = np.sqrt(max(R["V"][c, c], 0.0))
            b = (R["th"][k] - truth[k]) / s if s > 0 else 0.0
            qp = R["Qk"][k]["p"] if k in R["Qk"] else float("nan")
            print(f"{PNAMES[k]:>16} {truth[k]:7.3f} {R['th'][k]:8.3f} {s:7.3f} {b:8.2f}   {qp:.3g}")
        if R.get("frozen"):
            print(f"   frozen: {[PNAMES[k] for k in R['frozen']]}")
        print(f"   Q_split p={R['split']['p']:.3g}")
        for f in R.get("excised", []):
            print(f"   EXCISED (unknown-unknown candidate) {f['obs']}: [{f['lo']:.0f},{f['hi']:.0f}] "
                  f"{f['nbins']} bins mean pull {f['mean_pull']:+.1f}")
        for f in R["flags"]:
            print(f"   FLAG {f['obs']}: [{f['lo']:.0f},{f['hi']:.0f}] {f['nbins']} bins mean pull {f['mean_pull']:+.1f}")
        if not R["flags"]:
            print("   FLAGS: none (clean region)")

    np.savez(f"output/altgen/{LABEL}.npz", mode=MODE, inj=inj_desc, truth=truth, subset=subset,
             pnames=PNAMES,
             **{f"{m}_th": R["th"] for m, R in results.items()},
             **{f"{m}_V": R["V"] for m, R in results.items()},
             **{f"{m}_sub": np.array(R["sub"]) for m, R in results.items()},
             **{f"{m}_chi2data": R["chi2_data"] for m, R in results.items()})
    log(f"[out] output/altgen/{LABEL}.npz")
    log("done")


if __name__ == "__main__":
    main()
