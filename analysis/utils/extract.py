"""Generic ACHILLES hepmc -> observable bank extractor (reused across experiments).

Consolidates the per-channel extractors into ONE dispatch.  The hepmc parse + absolute-norm
boilerplate (analysis.utils.hepmc) is shared; each channel keeps its own
selection + observable logic verbatim.  cc0pi cuts are parametrized by --experiment (T2K | MINERvA).

Usage:
  python -m analysis.utils.extract <channel> <hepmc> [out.npz] [--experiment t2k|minerva] [--seed 0]
    channel : cc0pi | cc1pi | cc1pi_rich | res_w
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root (analysis/utils/ -> .)
from analysis.utils.hepmc import parse_events
from analysis.utils.hepmc import hepmc_norm

MU, NU_MU, PIP, PROT, NEUT = 13, 14, 211, 2212, 2112
PIONS = {111, 211, -211}
MESONS = {111, 211, -211, 221, 130, 310, 311, 321, -321, -311}
NUCLEONS = {2112, 2212}
M_A, M_A1 = 11174.862, 10252.547          # 12C, 11B [MeV]
COS70 = np.cos(70.0 * np.pi / 180.0)
COS20 = np.cos(20.0 * np.pi / 180.0)

# CC0pi muon/proton acceptance windows per experiment: (p_lo, p_hi|None, cos_lo).
CC0PI_CUTS = {
    "t2k":       dict(mu=(250.0, None, -0.6), prot=(450.0, 1000.0, 0.4)),       # arXiv:1802.05078
    "minerva":   dict(mu=(1500.0, 10000.0, COS20), prot=(450.0, 1200.0, COS70)),
    "microboone": dict(mu=(100.0, None, -1.0), prot=(300.0, 1200.0, -1.0)),     # ~4pi Ar acceptance (CC0pi-Np)
}


def _mom(p4):
    return np.sqrt(p4[1] ** 2 + p4[2] ** 2 + p4[3] ** 2)


# --------------------------------------------------------------------------- CC0pi-Np STV
def cc0pi(path, experiment="t2k", **_):
    cuts = CC0PI_CUTS[experiment]
    (mu_lo, mu_hi, mu_cos), (p_lo, p_hi, p_cos) = cuts["mu"], cuts["prot"]
    dpt, dat, q2, wv, pn, w, proc = [], [], [], [], [], [], []
    pmu_t, pmu_l, cmu_l, pmu_m = [], [], [], []
    for evt in parse_events(Path(path)):
        mu = None; protons = []; n_meson = 0; nu = None; pstr = None
        for pid, status, p4 in evt["parts"]:
            if pid == NU_MU and (nu is None or p4[0] > nu[0]):
                nu = np.asarray(p4)                 # beam neutrino (max energy)
            if status == 2 and pid in NUCLEONS and pstr is None:
                pstr = np.asarray(p4)               # struck (initial-state) nucleon for vertex W
            if status != 1:
                continue
            if pid == MU:
                mu = np.asarray(p4)
            elif pid in MESONS:
                n_meson += 1
            elif pid == PROT:
                protons.append(np.asarray(p4))
        if mu is None or n_meson != 0 or not protons:
            continue
        pmu = _mom(mu)
        if pmu < mu_lo or (mu_hi is not None and pmu > mu_hi) or mu[3] / pmu < mu_cos:
            continue
        # NUISANCE CC0pi STV: the HIGHEST-momentum proton must itself pass the window
        lead = max(protons, key=_mom); pl = _mom(lead)
        if not (p_lo < pl < p_hi and lead[3] / pl > p_cos):
            continue
        lt = mu[1:3]; pt = lead[1:3]; dvec = lt + pt; dptmag = float(np.linalg.norm(dvec))
        c = -np.dot(lt, dvec) / (np.linalg.norm(lt) * dptmag + 1e-9)
        dpt.append(dptmag); dat.append(float(np.arccos(np.clip(c, -1, 1))))
        qv = nu - mu if nu is not None else None
        q2.append(float((qv[1] ** 2 + qv[2] ** 2 + qv[3] ** 2 - qv[0] ** 2) / 1e6) if qv is not None else np.nan)
        if qv is not None and pstr is not None:
            tot = qv + pstr; wv.append(float(np.sqrt(max(tot[0] ** 2 - np.sum(tot[1:] ** 2), 0.0))))
        else:
            wv.append(np.nan)
        pL = mu[3] + lead[3]; Evis = mu[0] + lead[0]; R = M_A + pL - Evis
        dpL = 0.5 * R - (M_A1 ** 2 + dptmag ** 2) / (2.0 * max(R, 1.0))
        pn.append(float(np.sqrt(max(dptmag ** 2 + dpL ** 2, 0.0))))
        w.append(evt["w"] or 1.0); proc.append(evt["proc"] or 0)
        pmu_t.append(float(np.hypot(mu[1], mu[2]))); pmu_l.append(float(mu[3]))
        cmu_l.append(float(mu[3] / pmu)); pmu_m.append(float(pmu))
    return dict(dpt=np.array(dpt), dalphat=np.array(dat), Q2=np.array(q2), W=np.array(wv),
                pn=np.array(pn), w=np.array(w), proc=np.array(proc, np.int64),
                pmu=np.array(pmu_m), pmu_T=np.array(pmu_t), pmu_L=np.array(pmu_l), cos_mu=np.array(cmu_l))


# --------------------------------------------------------------------------- CC1pi+ STV
def cc1pi(path, seed=0, **_):
    """T2K CC1pi+ tight signal (PRD 103 112009): mu 250-7000, pi+ 150-1200, lead p 450-1200, theta<70."""
    MU_LO, MU_HI = 250.0, 7000.0; PI_LO, PI_HI = 150.0, 1200.0; P_LO, P_HI = 450.0, 1200.0

    def _in_accept(p4, lo, hi):
        m = _mom(p4)
        return lo < m < hi and (p4[3] / m) > COS70

    beam = np.array([0.0, 0.0, 1.0]); rng = np.random.default_rng(seed)
    dptt, pn, dat, dpt, w, ish = [], [], [], [], [], []
    pi_p, pi_cth, lp_p, Wv, Q2v, Enu = [], [], [], [], [], []
    for evt in parse_events(Path(path)):
        mu = pip = None; protons = []; n_other_meson = 0; n_pip = 0
        nu = None; pstr = None; struck_p = None
        for pid, status, p4 in evt["parts"]:
            if pid == NU_MU and (nu is None or p4[0] > nu[0]):
                nu = np.asarray(p4)
            if status == 2 and pid in NUCLEONS and pstr is None:
                pstr = np.asarray(p4); struck_p = _mom(p4)
            if status != 1:
                continue
            if pid == MU:
                mu = np.asarray(p4)
            elif pid == PIP:
                n_pip += 1; pip = np.asarray(p4)
            elif pid in MESONS:
                n_other_meson += 1
            elif pid == PROT:
                protons.append(np.asarray(p4))
        if mu is None or pip is None or n_pip != 1 or n_other_meson != 0 or not protons:
            continue
        if not (_in_accept(mu, MU_LO, MU_HI) and _in_accept(pip, PI_LO, PI_HI)):
            continue
        acc_p = [p for p in protons if _in_accept(p, P_LO, P_HI)]
        if not acc_p:
            continue
        p = max(acc_p, key=_mom); is_h = struck_p is not None and struck_p < 1.0
        if nu is not None and pstr is not None:
            q = nu - mu; tot = q + pstr
            Wv.append(float(np.sqrt(max(tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2 - tot[3] ** 2, 0.0))))
            Q2v.append(float((q[1] ** 2 + q[2] ** 2 + q[3] ** 2) - q[0] ** 2)); Enu.append(float(nu[0]))
        else:
            Wv.append(0.0); Q2v.append(0.0); Enu.append(0.0)
        ish.append(bool(is_h))
        pim = float(np.linalg.norm(pip[1:])); pi_p.append(pim); pi_cth.append(float(pip[3] / max(pim, 1e-9)))
        lp_p.append(float(np.linalg.norm(p[1:])))
        mu3, pi3, p3 = mu[1:], pip[1:], p[1:]
        zhat = np.cross(beam, mu3); zhat = zhat / (np.linalg.norm(zhat) + 1e-9)
        had3 = pi3 + p3; dptt_v = float(np.dot(had3, zhat)); dptt.append(dptt_v)
        lt = mu3[:2]; dpt_vec = lt + had3[:2]; dptmag = np.linalg.norm(dpt_vec); dpt.append(float(dptmag))
        if is_h or abs(dptt_v) < 0.5:
            dat.append(float(rng.uniform(0.0, np.pi)))      # NUISANCE hydrogen prescription
        else:
            c = -np.dot(lt, dpt_vec) / (np.linalg.norm(lt) * dptmag + 1e-9)
            dat.append(float(np.arccos(np.clip(c, -1, 1))))
        pL = mu[3] + pip[3] + p[3]; Evis = mu[0] + pip[0] + p[0]; R = M_A + pL - Evis
        dpL = 0.5 * R - (M_A1 ** 2 + dptmag ** 2) / (2.0 * max(R, 1.0))
        pn.append(float(np.sqrt(max(dptmag ** 2 + dpL ** 2, 0.0)))); w.append(evt["w"] or 1.0)
    return dict(dptt=np.array(dptt), pn=np.array(pn), dalphat=np.array(dat), dpt=np.array(dpt),
                w=np.array(w), is_h=(np.abs(np.array(dptt)) < 0.5), pi_p=np.array(pi_p),
                pi_cth=np.array(pi_cth), lp_p=np.array(lp_p), W=np.array(Wv), Q2=np.array(Q2v), Enu=np.array(Enu))


# --------------------------------------------------------------------------- CC1pi RICH (no cut)
def cc1pi_rich(path, K=4, M=10, **_):
    """Full final-state per event (NO signal cut -> re-bin offline): mu, pions (K, +pid),
    protons + neutrons (M each), struck (+pid), nu, n_other_meson, proc, weight."""
    mu_l, nu_l, st_l, sp_l, w_l, nom_l, proc_l = [], [], [], [], [], [], []
    pip4_l, pipid_l, pr4_l, nr4_l = [], [], [], []
    n_evt = nkept = 0
    for evt in parse_events(Path(path)):
        n_evt += 1
        mu = nu = struck = None; struck_pid = 0; pions = []; protons = []; neutrons = []; n_other = 0
        for pid, status, p4 in evt["parts"]:
            if pid == NU_MU and (nu is None or p4[0] > nu[0]):
                nu = p4
            if status == 2 and pid in NUCLEONS and struck is None:
                struck = p4; struck_pid = pid
            if status != 1:
                continue
            if pid == MU:
                mu = p4
            elif pid in PIONS:
                pions.append((pid, p4)); n_other += (pid != PIP)
            elif pid in MESONS:
                n_other += 1
            elif pid == PROT:
                protons.append(p4)
            elif pid == NEUT:
                neutrons.append(p4)
        if mu is None or nu is None or struck is None:
            continue
        nkept += 1
        p4p = np.zeros((K, 4)); pidp = np.zeros(K, np.int64)
        for i, (pp, q) in enumerate(sorted(pions, key=lambda t: -np.linalg.norm(t[1][1:]))[:K]):
            p4p[i] = q; pidp[i] = pp
        prp = np.zeros((M, 4))
        for i, q in enumerate(sorted(protons, key=lambda v: -np.linalg.norm(v[1:]))[:M]):
            prp[i] = q
        nrp = np.zeros((M, 4))
        for i, q in enumerate(sorted(neutrons, key=lambda v: -np.linalg.norm(v[1:]))[:M]):
            nrp[i] = q
        mu_l.append(mu); nu_l.append(nu); st_l.append(struck); sp_l.append(struck_pid)
        pip4_l.append(p4p); pipid_l.append(pidp); pr4_l.append(prp); nr4_l.append(nrp)
        nom_l.append(n_other); w_l.append(evt["w"] or 1.0)
        proc_l.append(evt["proc"] if evt["proc"] is not None else -1)
        if n_evt % 200000 == 0:
            print(f"  {n_evt} parsed, {nkept} kept", flush=True)
    return dict(mu=np.array(mu_l), nu=np.array(nu_l), struck=np.array(st_l), struck_pid=np.array(sp_l),
                pi_p4=np.array(pip4_l), pi_pid=np.array(pipid_l), prot_p4=np.array(pr4_l),
                neut_p4=np.array(nr4_l), n_other_meson=np.array(nom_l), w=np.array(w_l),
                proc=np.array(proc_l, np.int64))


# --------------------------------------------------------------------------- RES vertex-W (no-FSI)
def res_w(path, **_):
    """Unselected RES events (>=1 primary pion) from a no-FSI hepmc: vertex W, Q2, Enu, leading-pion
    kinematics, struck |p| (C/H split), full-signal in-window-proton flag."""
    W_l, Q2_l, E_l, pp_l, ps_l, w_l = [], [], [], [], [], []
    pcth_l, np_l, npc_l, mu_p_l, mu_c_l, ppid_l, npi_l, pok_l = [], [], [], [], [], [], [], []
    n_evt = 0
    for evt in parse_events(Path(path)):
        n_evt += 1
        nu = mu = pstr = None; pis = []; nucs = []; nuc_pid = []
        for pid, status, p4 in evt["parts"]:
            if pid == NU_MU and (nu is None or p4[0] > nu[0]):
                nu = np.asarray(p4)
            if status == 2 and pid in NUCLEONS and pstr is None:
                pstr = np.asarray(p4)
            if status != 1:
                continue
            if pid == MU:
                mu = np.asarray(p4)
            elif pid in PIONS:
                pis.append((pid, np.asarray(p4)))
            elif pid in NUCLEONS:
                nucs.append(np.asarray(p4)); nuc_pid.append((pid, np.asarray(p4)))
        if not pis or mu is None or nu is None or pstr is None:
            continue
        lpid, lpi = max(pis, key=lambda t: np.linalg.norm(t[1][1:]))
        lnuc = max(nucs, key=lambda v: np.linalg.norm(v[1:])) if nucs else np.zeros(4)
        npi_l.append(len(pis))
        prot_ok = any(pidn == 2212 and 450 < np.linalg.norm(p4n[1:]) < 1200
                      and p4n[3] / max(np.linalg.norm(p4n[1:]), 1e-9) > 0.342 for pidn, p4n in nuc_pid)
        pok_l.append(prot_ok)
        pim = np.linalg.norm(lpi[1:]); num_ = np.linalg.norm(lnuc[1:]); mum = np.linalg.norm(mu[1:])
        pcth_l.append(float(lpi[3] / max(pim, 1e-9))); ppid_l.append(int(lpid))
        np_l.append(float(num_)); npc_l.append(float(lnuc[3] / max(num_, 1e-9)))
        mu_p_l.append(float(mum)); mu_c_l.append(float(mu[3] / max(mum, 1e-9)))
        q = nu - mu; tot = q + pstr
        W_l.append(float(np.sqrt(max(tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2 - tot[3] ** 2, 0.0))))
        Q2_l.append(float(np.sum(q[1:] ** 2) - q[0] ** 2)); E_l.append(float(nu[0]))
        pp_l.append(float(np.linalg.norm(lpi[1:]))); ps_l.append(float(np.linalg.norm(pstr[1:])))
        w_l.append(evt["w"] or 1.0)
        if n_evt % 200000 == 0:
            print(f"  {n_evt} events parsed, {len(w_l)} RES", flush=True)
    return dict(W=np.array(W_l), Q2=np.array(Q2_l), Enu=np.array(E_l), pi_p=np.array(pp_l),
                pstr=np.array(ps_l), w=np.array(w_l), pi_cth=np.array(pcth_l), pi_pid=np.array(ppid_l),
                nuc_p=np.array(np_l), nuc_cth=np.array(npc_l), mu_p=np.array(mu_p_l), mu_cth=np.array(mu_c_l),
                n_pi=np.array(npi_l), prot_ok=np.array(pok_l))


def cc_incl(path, **_):
    """CC-inclusive: any charged-current muon event.  Muon kinematics only (pmu, cosθμ, pT, p||) --
    the MicroBooNE-style broad-acceptance observable.  No hadronic requirement."""
    pmu_m, cmu, pmu_t, pmu_l, enu, w, proc = [], [], [], [], [], [], []
    for evt in parse_events(Path(path)):
        mu = None; nu = None
        for pid, status, p4 in evt["parts"]:
            if pid == NU_MU and (nu is None or p4[0] > nu[0]):
                nu = np.asarray(p4)
            if status == 1 and pid == MU:
                mu = np.asarray(p4)
        if mu is None:
            continue
        pm = _mom(mu)
        pmu_m.append(float(pm)); cmu.append(float(mu[3] / pm))
        pmu_t.append(float(np.hypot(mu[1], mu[2]))); pmu_l.append(float(mu[3]))
        enu.append(float(nu[0]) if nu is not None else np.nan); w.append(evt["w"] or 1.0)
        proc.append(evt["proc"] if evt["proc"] is not None else -1)   # 200=QE, 401/402=RES
    return dict(pmu=np.array(pmu_m), cos_mu=np.array(cmu), pmu_T=np.array(pmu_t),
                pmu_L=np.array(pmu_l), Enu=np.array(enu), w=np.array(w), proc=np.array(proc, np.int64))


CHANNELS = {"cc0pi": cc0pi, "cc1pi": cc1pi, "cc1pi_rich": cc1pi_rich, "res_w": res_w, "cc_incl": cc_incl}


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACHILLES hepmc -> observable bank (T2K/MINERvA).")
    ap.add_argument("channel", choices=list(CHANNELS))
    ap.add_argument("hepmc")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--experiment", default="t2k", choices=list(CC0PI_CUTS))
    ap.add_argument("--seed", type=int, default=0)        # cc1pi hydrogen daT throw
    ap.add_argument("-K", type=int, default=4)            # cc1pi_rich pion pad
    ap.add_argument("-M", type=int, default=10)           # cc1pi_rich nucleon pad
    a = ap.parse_args(argv)
    out = a.out or f"output/achilles/{a.experiment}_{a.channel}.npz"
    d = CHANNELS[a.channel](a.hepmc, experiment=a.experiment, seed=a.seed, K=a.K, M=a.M)
    nrm = hepmc_norm(a.hepmc)                              # absolute scale from the hepmc header
    d.update(gen_xs_pb=nrm["gen_xs_pb"], sum_w_all=nrm["sum_w_all"], weight_to_nb=nrm["weight_to_nb"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    w = d["w"]; print(f"{a.channel} [{a.experiment}]: {len(w)} events  sum_w={w.sum():.4e} nb -> {out}", flush=True)


if __name__ == "__main__":
    main()
