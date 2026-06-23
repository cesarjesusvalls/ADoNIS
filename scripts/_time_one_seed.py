"""Time ONE production seed (cv5 params: P=10, n_recoil=2, mprot=4, max_steps=600, cylinder, pool) of RES
generation -> the usual CC1pi ADoNIS-vs-ACHILLES ratio+chi2 plot, with a live tqdm progress bar (the
cascade is chunked over events so progress is visible).  Reports per-stage wall time."""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "2"                            # cv5 n_recoil
import numpy as np, jax, jax.numpy as jnp, yaml
from tqdm import tqdm
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF

N, NCHUNK, MPROT = 30000, 12, 4
tg = resolve_targets("C")[0][0]
from adonis.xsec.spectral import SpectralFunction
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
CFG = DiscreteCascadeConfig(step=0.04, max_steps=600, seed=1, nn_inelastic=True, pauli=True,
                            early_exit=True, nucleus=tg.density_p, density_n=tg.density_n,
                            configs=tg.configs, engine="pool")   # GAUSSIAN (apples-to-apples vs ACHILLES-Gaussian)
T0 = time.time(); st = lambda m: print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

# ---- 1 seed of primary RES events ----
st("generating primary RES events (1 seed)...")
a = gen_events("res", N, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
m = len(a["w"]); st(f"primary done: {m} RES events -> cascading in {NCHUNK} chunks")

def one_chunk(sl, key):
    g = lambda k: jnp.asarray(a[k][sl])
    pterm, nterms, ofl, created = CF.cascade_nucleus(
        g("p_pi"), g("p_N"), g("ppid").astype(jnp.int32), g("ipid").astype(jnp.int32),
        g("Npid").astype(jnp.int32), CFG, key, P=10, max_gen=3, channel="res")
    p4s = []
    for gg in nterms:
        sp = np.asarray(gg["species"]); pid = np.asarray(gg["pid"]); p4 = np.asarray(gg["p4"]); al = np.asarray(gg["alive"])
        p4s.append(np.where(((sp == CF.NUCLEON) & (pid == 2212) & al)[:, :, None], p4, 0.0))
    P4 = np.concatenate(p4s, axis=1); mom = np.linalg.norm(P4[:, :, 1:], axis=2)
    idx = np.argsort(-mom, axis=1)[:, :MPROT]; ar = np.arange(P4.shape[0])[:, None]
    return dict(mu=np.asarray(a["k_mu"][sl]), nu=np.asarray(a["k_nu"][sl]), struck=np.asarray(a["p_struck"][sl]),
                pid_Ni=a["ipid"][sl].astype(np.int64), pi_post=np.asarray(pterm["p4"]),
                pid_pi=np.asarray(pterm["pid"]).astype(np.int64), pi_nsc=np.asarray(pterm["nsc"]).astype(np.int64),
                cr_p4=np.asarray(created["p4"]), cr_pid=np.where(np.asarray(created["alive"]), np.asarray(created["pid"]), 0).astype(np.int64),
                prot=P4[ar, idx], prot_origin=np.zeros((P4.shape[0], MPROT), np.int64), prot_gen=np.zeros((P4.shape[0], MPROT), np.int64),
                w=np.asarray(a["w"][sl]), ipid=a["ipid"][sl].astype(np.int64), Npid=a["Npid"][sl].astype(np.int64))

bounds = np.linspace(0, m, NCHUNK + 1).astype(int)
parts = []
tcasc = time.time()
for c in tqdm(range(NCHUNK), desc="cascade chunks (1st ~compile)", file=sys.stdout):
    parts.append(one_chunk(slice(bounds[c], bounds[c + 1]), jax.random.PRNGKey(11 + c)))
bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
st(f"cascade done ({time.time()-tcasc:.1f}s for {m} events)")
out = "data/oracle/t2k_cc1pi_engine_rich_1seed_gauss.npz"; np.savez(out, **bank); st(f"bank saved -> {out}")

# ---- usual CC1pi ratio+chi2 plot vs ACHILLES-C ----
st("building CC1pi ADoNIS-vs-ACHILLES ratio plot...")
from adonis.workflow.config import load_analysis_config
import adonis.workflow.analyze as A
from scripts.gen_cc_matrix import signal_block, banks, _obs
sigd, obs = signal_block("cc1pi", "res", "incl")
inp = banks("res", out, out); inp["reference"] = ["data/oracle/t2k_cc1pi_rich_ach_FSI_C_gauss.npz"]
cfg_d = dict(inputs=inp, signal=sigd, observables=_obs(obs), data={"enabled": False},
             out_path="paper_figures/cc1pi_res_incl_1seed_gauss.png", carbon_only=True, ratio_ylim=[0.5, 1.6],
             title="C cc1pi[res] incl 1-seed GAUSSIAN: ADoNIS vs ACHILLES (both Gaussian)")
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh: yaml.safe_dump(cfg_d, fh); tmp = fh.name
cfg = load_analysis_config(tmp); os.unlink(tmp); A.run_analysis(cfg)
st(f"DONE -> paper_figures/cc1pi_res_incl_1seed_gauss.png  (TOTAL {time.time()-T0:.1f}s)")
