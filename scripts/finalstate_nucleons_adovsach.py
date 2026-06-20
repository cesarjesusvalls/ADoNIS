"""ALL final-state nucleons (not just primaries) ADoNIS vs ACHILLES, carbon, protons & neutrons in
SEPARATE plots.  ACHILLES: every status==1 nucleon in the QE+RES fate hepmc (with per-event weight).
ADoNIS: every final-state proton/neutron from the FULL cascade (cascade_carbon_v2) over QE+RES, summed
with the absolute per-event weights so the QE:RES mix matches.  Momentum spectra, normalized, + ratio."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "6"
import numpy as np, jax, jax.numpy as jnp
np.seterr(all="ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF
from adonis.xsec.spectral import SpectralFunction
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"

HEPMC = sys.argv[1] if len(sys.argv) > 1 else "_oracle_out/T2K_C_fate.hepmc"
NEV = int(sys.argv[2]) if len(sys.argv) > 2 else 60000

# ---------- ACHILLES: parse final-state (status 1) nucleons + per-event weight ----------
ap, an, aw_p, aw_n = [], [], [], []
w_evt = 1.0
with open(HEPMC) as f:
    for ln in f:
        if ln[0] == "E":
            w_evt = None
        elif ln[0] == "W" and w_evt is None:                # first NUMERIC W after E = per-event CV weight
            t = ln.split()                                  # ('W CV' names line is non-numeric -> skip it)
            if len(t) >= 2:
                try: w_evt = float(t[1])
                except ValueError: pass
        elif ln[0] == "P":
            t = ln.split()
            pid = int(t[3]); st = int(t[-1])
            if st == 1 and pid in (2212, 2112):
                p = np.sqrt(float(t[4])**2 + float(t[5])**2 + float(t[6])**2)
                if pid == 2212: ap.append(p); aw_p.append(w_evt or 1.0)
                else: an.append(p); aw_n.append(w_evt or 1.0)
ap = np.array(ap); an = np.array(an); aw_p = np.array(aw_p); aw_n = np.array(aw_n)
print(f"[ACHILLES] final-state p={len(ap)} n={len(an)}", flush=True)

# ---------- ADoNIS: full cascade QE+RES, every final-state p/n ----------
tg = resolve_targets("C")[0][0]
cfg = DiscreteCascadeConfig(cylinder=True, step=0.04, max_steps=600, seed=1, nn_inelastic=True, engine="pool",
                            pauli=True, nucleus=tg.density_p, density_n=tg.density_n, configs=tg.configs)
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
dp, dn, dwp, dwn = [], [], [], []
for ch in ("qe", "res"):
    a = gen_events(ch, NEV, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
    w = np.asarray(a["w"]); s = w > 0; m = int(s.sum()); w = w[s]
    p_pi = jnp.asarray(a["p_pi"][s]) if "p_pi" in a else jnp.asarray(a["p_N"][s])
    pterm, nterms, ofl, created = CF.cascade_carbon_v2(
        p_pi, jnp.asarray(a["p_N"][s]), jnp.asarray(a["ppid"][s], jnp.int32),
        jnp.asarray(a["ipid"][s], jnp.int32), jnp.asarray(a["Npid"][s], jnp.int32),
        cfg, jax.random.PRNGKey(11), P=12, max_gen=6, channel=ch)
    for g in nterms:
        sp = np.asarray(g["species"]); pid = np.asarray(g["pid"]); p4 = np.asarray(g["p4"]); al = np.asarray(g["alive"])
        mom = np.linalg.norm(p4[:, :, 1:], axis=2)
        # EXCLUDE recaptured/captured nucleons (recap zeroes |p|->0 = bound), matching ACHILLES which
        # excludes status-26 (captured) from the final state (status 1).  Final-state = ESCAPED only.
        isN = (sp == CF.NUCLEON) & al & (mom > 1.0)
        wp = np.broadcast_to(w[:, None], mom.shape)
        mp = isN & (pid == 2212); mn = isN & (pid == 2112)
        dp.append(mom[mp]); dwp.append(wp[mp]); dn.append(mom[mn]); dwn.append(wp[mn])
    print(f"[ADoNIS {ch}] events={m}", flush=True)
dp = np.concatenate(dp); dn = np.concatenate(dn); dwp = np.concatenate(dwp); dwn = np.concatenate(dwn)
np.savez("/tmp/fs_nuc_arrays.npz", dp=dp, dn=dn, dwp=dwp, dwn=dwn, ap=ap, an=an, aw_p=aw_p, aw_n=aw_n)
print(f"[ADoNIS] final-state p={len(dp)} n={len(dn)}  (cached arrays -> /tmp/fs_nuc_arrays.npz)", flush=True)

# ---------- plot: proton + neutron momentum spectra, normalized, ADoNIS vs ACHILLES ----------
edges = np.linspace(0, 1200, 31); ctr = 0.5*(edges[:-1]+edges[1:])
def norm_hist(x, w): h = np.histogram(x, bins=edges, weights=w)[0]; return h/h.sum()
for sp, (xd, wd, xa, wa) in [("proton", (dp, dwp, ap, aw_p)), ("neutron", (dn, dwn, an, aw_n))]:
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    hd = norm_hist(xd, wd); ha = norm_hist(xa, wa)
    ax[0].step(ctr, hd, where="mid", color="crimson", lw=2, label="ADoNIS")
    ax[0].step(ctr, ha, where="mid", color="navy", lw=2, label="ACHILLES")
    ax[0].set_ylabel("normalized"); ax[0].legend(); ax[0].set_title(f"Carbon: ALL final-state {sp}s  |p|  (ADoNIS vs ACHILLES)")
    ax[1].step(ctr, hd/np.clip(ha, 1e-12, None), where="mid", color="k"); ax[1].axhline(1, ls=":", c="gray")
    ax[1].set_ylabel("ADO/ACH"); ax[1].set_xlabel("$|p|$ [MeV]"); ax[1].set_ylim(0.5, 1.5)
    plt.tight_layout(); out = f"paper_figures/finalstate_{sp}_C.png"; plt.savefig(out, dpi=120); print("wrote", out)
