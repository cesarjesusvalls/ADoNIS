"""ADoNIS 1-seed GAUSSIAN QE bank -> CC0pi QE-only ADoNIS-vs-ACHILLES ratio+chi2 plot (vs the Gaussian
ACHILLES ref, proc 200).  tqdm bar over cascade chunks; reports per-stage wall time.  ACHILLES QE ref is
FREE -- the same Gaussian card generates QE (proc 200) + RES, already in t2k_cc1pi_rich_ach_FSI_C_gauss.npz."""
import os, sys, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ADONIS_N_RECOIL"] = "2"
import numpy as np, jax, jax.numpy as jnp, yaml
from tqdm import tqdm
import adonis.xsec.flux as flux; flux.BEAM_MODE = "is"
from adonis.workflow.materials import resolve_targets
from adonis.workflow.generate import gen_events
from adonis.fsi.cascade_discrete import DiscreteCascadeConfig
from adonis.fsi import cascade_full as CF
from adonis.xsec.spectral import SpectralFunction

N, NCHUNK, MPROT = 30000, 12, 4
tg = resolve_targets("C")[0][0]
sf_n = SpectralFunction(tg.spectral_n); sf_p = SpectralFunction(tg.spectral_p)
CFG = DiscreteCascadeConfig(step=0.04, max_steps=600, seed=1, nn_inelastic=True, pauli=True,
                            early_exit=True, nucleus=tg.density_p, density_n=tg.density_n,
                            configs=tg.configs, engine="pool")    # GAUSSIAN
T0 = time.time(); st = lambda m: print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

st("generating primary QE events (1 seed)...")
a = gen_events("qe", N, 0, sf_n=sf_n, sf_p=sf_p, n_neutron=tg.A - tg.Z, n_proton=tg.Z)
m = len(a["w"]); st(f"primary done: {m} QE events -> cascading in {NCHUNK} chunks")

def one_chunk(sl, key):
    g = lambda k: jnp.asarray(a[k][sl])
    p_pi_dummy = jnp.zeros((int((sl.stop or m) - (sl.start or 0)), 4))
    pterm, nterms, ofl, created = CF.cascade_nucleus(
        p_pi_dummy, g("p_N"), g("ppid").astype(jnp.int32), g("ipid").astype(jnp.int32),
        g("Npid").astype(jnp.int32), CFG, key, P=10, channel="qe")
    p4s = []
    for gg in nterms:
        sp = np.asarray(gg["species"]); pid = np.asarray(gg["pid"]); p4 = np.asarray(gg["p4"]); al = np.asarray(gg["alive"])
        p4s.append(np.where(((sp == CF.NUCLEON) & (pid == 2212) & al)[:, :, None], p4, 0.0))
    P4 = np.concatenate(p4s, axis=1); mom = np.linalg.norm(P4[:, :, 1:], axis=2)
    idx = np.argsort(-mom, axis=1)[:, :MPROT]; ar = np.arange(P4.shape[0])[:, None]
    return dict(mu=np.asarray(a["k_mu"][sl]), nu=np.asarray(a["k_nu"][sl]), struck=np.asarray(a["p_struck"][sl]),
                pid_Ni=a["ipid"][sl].astype(np.int64), pi_post=np.asarray(pterm["p4"]),
                pid_pi=np.asarray(pterm["pid"]).astype(np.int64), pi_nsc=np.zeros(P4.shape[0], np.int64),
                cr_p4=np.asarray(created["p4"]), cr_pid=np.where(np.asarray(created["alive"]), np.asarray(created["pid"]), 0).astype(np.int64),
                prot=P4[ar, idx], prot_origin=np.zeros((P4.shape[0], MPROT), np.int64), prot_gen=np.zeros((P4.shape[0], MPROT), np.int64),
                w=np.asarray(a["w"][sl]), ipid=a["ipid"][sl].astype(np.int64), Npid=a["Npid"][sl].astype(np.int64))

bounds = np.linspace(0, m, NCHUNK + 1).astype(int); parts = []; tc = time.time()
for c in tqdm(range(NCHUNK), desc="QE cascade chunks (1st ~compile)", file=sys.stdout):
    parts.append(one_chunk(slice(bounds[c], bounds[c + 1]), jax.random.PRNGKey(11 + c)))
bank = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
st(f"cascade done ({time.time()-tc:.1f}s for {m} events)")
out = "data/oracle/t2k_cc0pi_1seed_gauss.npz"; np.savez(out, **bank); st(f"bank saved -> {out}")

st("building CC0pi QE-only ADoNIS-vs-ACHILLES ratio plot...")
from adonis.workflow.config import load_analysis_config
import adonis.workflow.analyze as A
from scripts.gen_cc_matrix import signal_block, banks, _obs
sigd, obs = signal_block("cc0pi", "qe", "incl")
inp = banks("qe", out, out); inp["reference"] = ["data/oracle/t2k_cc1pi_rich_ach_FSI_C_gauss.npz"]
cfg_d = dict(inputs=inp, signal=sigd, observables=_obs(obs), data={"enabled": False},
             out_path="paper_figures/cc0pi_qe_incl_1seed_gauss.png", carbon_only=True, ratio_ylim=[0.5, 1.6],
             title="C cc0pi[qe] incl 1-seed GAUSSIAN: ADoNIS vs ACHILLES (both Gaussian)")
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh: yaml.safe_dump(cfg_d, fh); tmp = fh.name
cfg = load_analysis_config(tmp); os.unlink(tmp); A.run_analysis(cfg)
st(f"DONE -> paper_figures/cc0pi_qe_incl_1seed_gauss.png  (TOTAL {time.time()-T0:.1f}s)")
