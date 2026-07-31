"""Generate the ADoNIS methods & figure report (output/paper/methods_report.html).

A REVIEW document, not a paper draft: for each paper figure, exactly how it is computed and how it
connects to the shared machinery.  Figures are embedded (downsized) as data URIs so the file is
self-contained.  Text is deliberately terse -- calculation chains, not motivation.

Usage:  python -m analysis.paper.methods_report   (then publish output/paper/methods_report.html)
"""
import base64
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
DEST = OUT / "paper" / "methods_report.html"
MAXW = 1150


def fig(relpath, w=MAXW):
    p = OUT / relpath
    if not p.exists():
        return f'<div class="missing">missing: {relpath}</div>'
    im = Image.open(p).convert("RGB")
    if im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="{Path(relpath).stem}" src="data:image/png;base64,{b64}">'


def figure(relpath, caption, w=MAXW):
    return f'<figure>{fig(relpath, w)}<figcaption>{caption}</figcaption></figure>'


CSS = """
:root{
  --bg:#f7f9fb; --panel:#ffffff; --ink:#16202b; --muted:#5a6b7a; --faint:#8595a4;
  --line:#e2e8ef; --line2:#eef2f6; --ado:#1f77b4; --ach:#d62728; --good:#2f8f4e; --warn:#b9770a;
  --code-bg:#f0f3f7; --chip:#eaf1f8;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0d1218; --panel:#151d27; --ink:#e6edf3; --muted:#9fb0c0; --faint:#6b7d8d;
  --line:#243040; --line2:#1b2530; --ado:#4a9fd8; --ach:#e5645f; --good:#4bb572; --warn:#d69524;
  --code-bg:#1a2431; --chip:#182430;
}}
:root[data-theme="light"]{
  --bg:#f7f9fb; --panel:#ffffff; --ink:#16202b; --muted:#5a6b7a; --faint:#8595a4;
  --line:#e2e8ef; --line2:#eef2f6; --ado:#1f77b4; --ach:#d62728; --good:#2f8f4e; --warn:#b9770a;
  --code-bg:#f0f3f7; --chip:#eaf1f8;
}
:root[data-theme="dark"]{
  --bg:#0d1218; --panel:#151d27; --ink:#e6edf3; --muted:#9fb0c0; --faint:#6b7d8d;
  --line:#243040; --line2:#1b2530; --ado:#4a9fd8; --ach:#e5645f; --good:#4bb572; --warn:#d69524;
  --code-bg:#1a2431; --chip:#182430;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:400 16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums;}
.wrap{max-width:1220px;margin:0 auto;padding:48px 28px 100px;}
.reading{max-width:74ch;}
code,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.86em;}
code{background:var(--code-bg);padding:.08em .38em;border-radius:4px;color:var(--ink);}
h1{font-size:30px;line-height:1.15;letter-spacing:-.01em;margin:0 0 8px;text-wrap:balance;}
.sub{color:var(--muted);font-size:16px;margin:0 0 4px;max-width:74ch;}
.meta{color:var(--faint);font-size:13px;margin:2px 0 0;}
h2{font-size:22px;letter-spacing:-.01em;margin:0;text-wrap:balance;}
h3{font-size:16px;margin:26px 0 6px;color:var(--ink);}
.sec{margin-top:56px;padding-top:30px;border-top:1px solid var(--line);}
.sechead{display:flex;align-items:baseline;gap:14px;margin-bottom:6px;}
.num{font:600 13px/1 ui-monospace,monospace;color:var(--ado);border:1px solid var(--line);
  background:var(--chip);border-radius:6px;padding:6px 8px;letter-spacing:.02em;white-space:nowrap;}
.tag{color:var(--muted);font-size:13.5px;margin:0 0 18px;}
p{margin:10px 0;}
.reading p, .sec > p{max-width:74ch;}
ul{max-width:74ch;padding-left:20px;margin:8px 0;}
li{margin:4px 0;}
b,strong{font-weight:650;}
.k{color:var(--ado);font-weight:600;}
figure{margin:22px 0 6px;background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:14px;overflow-x:auto;}
figure img{display:block;width:100%;height:auto;border-radius:4px;}
figcaption{color:var(--muted);font-size:13px;margin-top:10px;line-height:1.5;max-width:none;}
.eyebrow{text-transform:uppercase;letter-spacing:.09em;font-size:11.5px;font-weight:650;color:var(--faint);
  margin:0 0 14px;}
.spine{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:18px 0 6px;}
.node{flex:1 1 130px;min-width:120px;background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:11px 12px;position:relative;}
.node .t{font:600 12px/1.3 ui-monospace,monospace;color:var(--ado);}
.node .d{font-size:12px;color:var(--muted);margin-top:3px;line-height:1.4;}
.node.term{border-color:var(--ado);}
.arrow{align-self:center;color:var(--faint);font-size:18px;flex:0 0 auto;}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:14px 0;}
th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--line2);vertical-align:top;}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint);font-weight:650;
  border-bottom:1px solid var(--line);}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap;}
.wrapscroll{overflow-x:auto;}
.chip{display:inline-block;font-size:11.5px;font-weight:600;padding:2px 8px;border-radius:20px;
  border:1px solid transparent;white-space:nowrap;}
.chip.ok{color:var(--good);background:color-mix(in srgb,var(--good) 12%,transparent);
  border-color:color-mix(in srgb,var(--good) 30%,transparent);}
.chip.gap{color:var(--warn);background:color-mix(in srgb,var(--warn) 12%,transparent);
  border-color:color-mix(in srgb,var(--warn) 30%,transparent);}
.callout{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--ado);
  border-radius:8px;padding:12px 16px;margin:16px 0;font-size:14px;max-width:74ch;}
.callout.warn{border-left-color:var(--warn);}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px 22px;margin:10px 0;}
@media(max-width:720px){.grid2{grid-template-columns:1fr;}}
.deffile{color:var(--faint);font-size:12.5px;margin-top:4px;}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin:6px 0 0;}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:5px;}
"""


def spine():
    nodes = [
        ("frozen bank", "QE+RES sampled once from the bit-exact generators; weights in absolute nb (18.3M ev)"),
        ("θ-indep. walk", "pool cascade at nominal → ragged kind-1 FSI records + amps² (a,b,c) hard-vertex records"),
        ("reweight w(θ)", "w₀·norm·hv·fsi·sf; =1 at nominal, exact at any θ; JAX-differentiable"),
        ("Jacobian J", "one jax.jvp per knob through weight_jit → 27 knobs × 188 bins"),
        ("sections", "§2 plots J · §3 Fisher F=JᵀC⁻¹J → shrinkage · beams add rows F+=F_beam"),
    ]
    parts = []
    for i, (t, d) in enumerate(nodes):
        cls = "node term" if i == len(nodes) - 1 else "node"
        parts.append(f'<div class="{cls}"><div class="t">{t}</div><div class="d">{d}</div></div>')
        if i < len(nodes) - 1:
            parts.append('<div class="arrow">→</div>')
    return '<div class="spine">' + "".join(parts) + "</div>"


READINESS = [
    ("1", "Validation: ADoNIS reproduces ACHILLES",
     "fig01_ee_domega · fig02_anl_sigma · fig03_pi_nucleus_sigma · fig456_e4nu · fig07/08/09 · fig10_uboone · fig13_piN_sigma",
     "ok", "figure-complete",
     "all non-NC ACHILLES-paper figures; ratios within a few % (near-threshold outliers documented: fig2 pπ⁺, fig13 ηN cusp)"),
    ("2", "Gradient information for all 27 knobs", "sec2_gradients_shape · sec2_gradient_reach",
     "ok", "figure-complete", "exact per-bin ∂(dσ/dx)/∂θ, autodiff, all 27"),
    ("3", "Fisher information per data subset",
     "sec3_shrinkage_subsets · sec3_failure_modes · sec3_degeneracy", "ok", "figure-complete",
     "T2K 10/27 FIT; knob×sample table (no figure yet)"),
    ("4", "Closures on the Fisher-selected knobs", "physfit_fig7_closure5", "ok", "figure-complete",
     "5-param recovered ≤ 0.5σ"),
    ("5", "Fitting data the model can't describe", "physfit_fig3/4/8/9/10", "ok", "figure-complete",
     "inside/outside-manifold taxonomy demonstrated"),
]


def readiness_table():
    rows = []
    for n, name, figs, st, lab, metric in READINESS:
        chip = f'<span class="chip {st}">{lab}</span>'
        rows.append(f'<tr><td class="n">{n}</td><td>{name}<div class="deffile mono">{figs}</div></td>'
                    f'<td>{chip}</td><td>{metric}</td></tr>')
    return ('<div class="wrapscroll"><table><thead><tr><th>§</th><th>Section / figures</th>'
            '<th>Status</th><th>Headline</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


BODY = f"""
<title>ADoNIS — Methods &amp; Figure Report</title>
<div class="wrap">

<header>
  <p class="eyebrow">Readiness review · not a paper draft</p>
  <h1>ADoNIS — Methods &amp; Figure Report</h1>
  <p class="sub">How every paper figure is computed, and how they connect through one shared chain.
  Terse by intent: calculation, not motivation.</p>
  <p class="meta">ADoNIS = differentiable (JAX) port of ACHILLES. Target 12C, T2K flux. Comparison metric is
  always ADoNIS vs ACHILLES.</p>
</header>

<section class="sec" style="border-top:none;padding-top:28px;margin-top:30px;">
  <div class="sechead"><span class="num">§0</span><h2>The shared chain</h2></div>
  <p class="tag">Everything below hangs off one object. Read this first; the sections are views of it.</p>
  {spine()}
  <div class="reading">
  <h3>Frozen absolute proposal</h3>
  <p>QE and RES events are sampled <b>once</b> from the bit-exact ACHILLES-faithful generators
  (<code>adonis/xsec</code>: spectral function, amps², flux). Weights <code>w₀</code> are absolute nb —
  no fitted normalization anywhere. The frozen set is <code>output/event_bank_v2</code> (18,276,613 ev =
  9.75M QE + 8.53M RES; 40 seed-sharded SLURM jobs, merged).</p>

  <h3>θ-independent walk + records</h3>
  <p>The pool cascade (<code>adonis/fsi/cascade_full.run_cascade_pool</code>) runs at nominal cross
  sections. It emits a per-event <b>kind-1 FSI record</b> — the exact sufficient statistics for the FSI
  reweight — stored <b>ragged</b> (flat per-slot arrays + a per-slot event index, like the ragged final
  state). Each pion slot carries the struck-candidate σ decomposition <em>and</em> the closest-candidate
  <code>a=πb²/σ_tot</code> for the hit/no-hit survival factor; each nucleon slot the analogous stats. The
  hard vertex stores amps²(θ) as quadratic <code>(a,b,c)</code> records per form-factor / M_A knob.</p>

  <h3>Exact reweight <span class="mono">w(θ)</span></h3>
  <p><code>bank_weight = w₀ · norm · hv · fsi · sf</code>
  (<code>analysis/t2k/differentiability/bank_reweight.py</code>): <code>hv</code> = amps² ratio from the
  <code>(a,b,c)</code> records (M_A dipole ratio / strength scale), <code>fsi</code> =
  <code>pool_fsi_reweight</code> (branch × survival × NN-cex, exp(Σ log) over ragged slots),
  <code>sf</code> = spectral-function density ratio. <b>= 1 at nominal exactly</b>, exact at any θ, and
  <code>jax.jit</code>-compiled + differentiable.</p>

  <h3>Jacobian <span class="mono">J</span></h3>
  <p><code>J_ik = ∂(dσ/dx)_i/∂θ_k</code> = one <code>jax.jvp</code> per knob through <code>weight_jit</code>,
  binned per observable. <b>27 knobs × 188 bins.</b> Persisted as
  <code>output/altgen/physfit_gate1_full_v2.npz</code>. §2 <em>plots</em> this J; §3 forms the Fisher from
  it. Every beam sample is just <b>extra rows of J</b> (Fisher is additive).</p>
  </div>
</section>

<section class="sec" style="border-top:none;padding-top:0;margin-top:40px;">
  <div class="sechead"><h2>Readiness at a glance</h2></div>
  {readiness_table()}
  <p class="tag">All five arc sections are figure-complete. Open items are listed at the end.</p>
</section>

<section class="sec">
  <div class="sechead"><span class="num">§1</span><h2>Validation — ADoNIS reproduces ACHILLES</h2></div>
  <p class="tag">Same source as Gate I, so this validates precisely the datasets §3 measures.</p>
  <div class="reading">
  <p><b>What.</b> Overlay of dσ/dx + a ratio panel + χ²/ndf, for all 11 neutrino observables (4 CC0π
  kinematic, 5 CC1π kinematic, 2 CC-inclusive multiplicity) and the 3 tagged beams.</p>
  <p><b>ADoNIS side.</b> The frozen event bank, via <code>build_physfit_datasets</code> — the same 11
  datasets and bin edges Gate I uses. Per-event observable values (STV formulas
  <code>bank_plot.dpt/dat/pN_1pi/dptt_1pi/dat_1pi</code>) + weight <code>w₀</code>, masked by
  <code>signal_cc0pi(topological)</code> / <code>signal_cc1pi_stv</code> / muon-acceptance.</p>
  <p><b>ACHILLES side.</b> The rich oracle <code>t2k_cc1pi_rich_ach_FSI_proc.npz</code> (10M ev, 50 independent seeds). The
  <b>same STV formulas</b> are applied to the oracle's own 4-vectors (muon, leading proton, π⁺); only the
  per-event extraction differs (bank ragged final state vs the oracle's top-K padded arrays). Weight =
  <code>w · weight_to_nb</code> (= w here; already absolute nb).</p>
  <p><b>Beams.</b> ADoNIS beam banks fired in ACHILLES's <code>InitCrossSection</code> geometry (beam
  uniform-|p|, impact parameter uniform in a 10 fm disk, z₀ = −1.05 R_nuc, external_test);
  <code>σ(p) = πR² · Σ_bin reacted·w(θ) / n_tried</code>. Reaction counted <b>post-Pauli</b> (per-particle
  <code>nsc</code>, never the pre-Pauli hit flag). ACHILLES from the cascade oracles.</p>
  <p><b>Plotter.</b> All neutrino panels go through the one validated
  <code>adonis/workflow/plotting.chi2_ratio_panel</code>. χ² over bins where both sides &gt; 0, stat
  errors √(Σw²)/binwidth.</p>
  <p class="deffile mono">analysis/paper/sec1_validation/make.py</p>
  </div>
  <div class="callout warn"><b>Selection note.</b> CC0π signal is <b>topological</b> (no final-state pion)
  on both sides — the only definition the oracle can express, and what an experiment measures. The same
  SignalDef-driven <code>bank_signal</code>/<code>oracle_signal</code> selection + validated STV formulas
  run on the ADoNIS paper_banks bank and the ACHILLES fs_rich oracle.</div>
  {figure("paper/fig01_ee_domega.png", "<b>Fig 1 — inclusive (e,e') dσ/dω.</b> ⁴⁰Ar & ¹²C @2.222 GeV, θ_e'≈15.5°, QE/RES/total; ADoNIS (solid) vs ACHILLES (dashed). χ²/ndf 1.04 / 1.56.")}
  {figure("paper/fig02_anl_sigma.png", "<b>Fig 2 — free-nucleon RES σ(E_ν).</b> pπ⁺ / nπ⁺ / pπ⁰; ADoNIS monochromatic free-nucleon evaluator vs the ACHILLES scan. χ²/ndf 32.8 / 3.4 / 0.9 (pπ⁺ dominated by the near-threshold E=400 point).")}
  {figure("paper/fig03_pi_nucleus_sigma.png", "<b>Fig 3 — π⁺ nucleus σ(p).</b> absorption + reaction on ¹²C & ⁴⁰Ar (ADoNIS = Virtual Resonances). χ²/ndf 0.58–1.3.")}
  {figure("paper/fig456_e4nu.png", "<b>Figs 4/5/6 — e4ν (e,e') on ¹²C @1.159 GeV.</b> E_QE (0π), E_cal (1p0π), P_T (1p0π). χ²/ndf 1.83 / 2.14 / 1.16.")}
  {figure("paper/fig07_t2k_cc0pi.png", "<b>Fig 7 — T2K CC0π TKI.</b> δp_T + δα_T (+ p_μ, cosθ_μ) on ¹²C. χ²/ndf 0.7–1.7.")}
  {figure("paper/fig08_t2k_cc1pi.png", "<b>Fig 8 — T2K CC1π⁺ STV.</b> p_N + δp_TT (+ δα_T, p_π, cosθ_π); CC1π+Np acceptance (momentum windows + cosθ&gt;cos70° on μ/π/leading-p).")}
  {figure("paper/fig09_minerva_cc0pi.png", "<b>Fig 9 — MINERvA CC0π TKI.</b> δα_T + p_n^reco (+ δp_T) on ¹²C.")}
  {figure("paper/fig10_uboone_cc1p0pi.png", "<b>Fig 10 — MicroBooNE CC1p0π.</b> double-differential δp_T in four δα_T slices on ⁴⁰Ar.")}
  {figure("paper/fig13_piN_sigma.png", "<b>Fig 13 — meson-baryon DCC.</b> total σ(W) off proton (pi+/pi0/pi-/eta) + angular at p=300 MeV; ADoNIS Monte-Carlo cascade sampler vs ANL-Osaka analytic.")}
</section>

<section class="sec">
  <div class="sechead"><span class="num">§2</span><h2>Gradient information for all 27 knobs</h2></div>
  <p class="tag">The object is the shared J; nothing new is computed.</p>
  <div class="reading">
  <p><b>What.</b> The full Jacobian as the per-bin <em>pull</em>
  <code>S_ik = σ_prior_k · J_ik / σ_i</code> — the dimensionless quantity the Fisher accumulates
  (<code>F = SᵀS</code>) — shown per-knob normalized to each knob's own peak, so WHERE each knob pulls
  across the bins is readable regardless of its magnitude. A row with no colour is a knob with no gradient.</p>
  <p><b>How.</b> Each row normalized to its own peak pull (magnitude removed; the RES knobs light the CC1π
  columns and reach CC0π via pion absorption). The companion reach bar is <code>√F_kk</code> = the total
  pull a 1σ prior move produces, degeneracy-blind; row label red = passes Gate I.</p>
  <p class="deffile mono">analysis/paper/sec2_gradients/make.py — reads physfit_gate1_full_v2.npz (no bank pass)</p>
  </div>
  {figure("paper/sec2_gradients_shape.png", "<b>Per-knob gradient shape, all 27 knobs × 188 bins</b>, autodiff through the cascade — each knob's row normalized to its own peak, so WHERE it pulls is readable regardless of magnitude. Columns grouped by observable. s_conv is blank (no gradient); the RES knobs light the CC1π columns and reach CC0π via pion absorption.")}
  {figure("paper/sec2_gradient_reach.png", "<b>Gradient reach √F_kk per knob.</b> Every knob has a gradient; not every gradient is information. Red = passes Gate I on the full set.")}
</section>

<section class="sec">
  <div class="sechead"><span class="num">§3</span><h2>Fisher information per observable subset</h2></div>
  <p class="tag">What is worth fitting, and why not — decided from the data.</p>
  <div class="reading">
  <p><b>What.</b> Gate I asks, per observable subset: does the <em>data</em> (not the prior) determine a
  knob? Asimov posterior <code>V = (JᵀC⁻¹J + Π⁻¹)⁻¹</code>; a knob is <b>FIT</b> when shrinkage
  <code>σ_post/σ_prior &lt; 0.5</code>. Diagonal error <code>σ_i² = (5%·dσ_i)² + MC²</code>.</p>
  <p><b>Two axes.</b> σ_post is <em>marginalized</em> (a diagonal of V), so failure has two opposite
  causes, reported separately: <span class="k">DEGENERATE</span> (raw <code>1/√F_kk</code> small — data
  sees it — but marginalized large: another knob mimics it; a better observable recovers it) vs
  <span class="k">INVISIBLE</span> (raw &gt; 1: no information at this precision; nothing recovers it).</p>
  <p><b>Subsets.</b> Every subset is a <b>row slice</b> of the one persisted J (no bank pass). lepton →
  initial state only (2 FIT); +pion-kin → +RES vertex; TKI → +nucleon FSI; +multiplicities → +FSI. Full
  11-observable set: <b>10/27 FIT</b>.</p>
  <p><b>Precision-relative.</b> At 15% systematic only 4 knobs pass instead of 10 — measurability is a
  property of the knob <em>and</em> the dataset's precision.</p>
  <p class="deffile mono">analysis/paper/physical_fit.py (Gate I) · analysis/paper/sec3_fisher/make.py</p>
  </div>
  {figure("paper/sec3_shrinkage_subsets.png", "<b>Shrinkage per probe type</b> (ν = T2K+MINERvA, e beam, hadron beam), marginalized (left) vs raw (right). Red box = FIT (fit measures it); green box = data sees it with other knobs fixed. Green-without-red = DEGENERATE.")}
  {figure("paper/sec3_failure_modes.png", "<b>The two failure modes.</b> x = can the data see it (raw), y = can the fit deliver it (marginalized). qe_norm is the extreme: raw 0.033 (one of the most sensitive knobs) → marginalized 0.86, a 26× degeneracy penalty against axial/vector_strength + sf_norm.")}
  {figure("paper/sec3_degeneracy.png", "<b>Degeneracy structure</b> — eigen-spectrum of the prior-scaled Fisher (left) and the eigenvector composition of the best-measured modes (right).")}
  <div class="reading">
  <h3>Knob × sample (the tagged-beam payoff)</h3>
  <p>Fisher is additive: <code>F = F_T2K + Σ_beam F_beam</code>, each beam just extra rows of J
  (<code>analysis/paper/beams/beam_fisher.py</code>). Each beam is <b>pure FSI</b> (no competing hard-vertex/SF
  knobs) and its <b>energy opens a channel shut at T2K</b>. Each rescues exactly its target knob:</p>
  </div>
  <div class="wrapscroll"><table><thead><tr><th>sample</th><th>#FIT</th><th>knob gained vs T2K</th><th>mechanism</th></tr></thead><tbody>
  <tr><td>T2K alone</td><td class="n">10</td><td>—</td><td></td></tr>
  <tr><td>+ π⁺–C</td><td class="n">11</td><td class="mono">s_conv</td><td>p_π→1 GeV/c opens πN→ηN′</td></tr>
  <tr><td>+ p–C</td><td class="n">11</td><td class="mono">s_NN_inelastic[pp]</td><td>KE&gt;290 MeV opens NN→NNπ</td></tr>
  <tr><td>+ n–C</td><td class="n">12</td><td class="mono">s_NN_elastic[nn], [nn]inel</td><td>the nn isospin channel</td></tr>
  <tr><td><b>ALL</b></td><td class="n"><b>14</b></td><td>+ all above; sabs 0.38→0.13</td><td>s_piN_elastic 0.42→0.16</td></tr>
  </tbody></table></div>
  <div class="callout warn"><b>Gap.</b> This knob×sample result is a table + npz (<code>beam_fisher.npz</code>);
  it has <b>no figure yet</b>. A knob×sample shrinkage heatmap is the one missing §3 figure.</div>
</section>

<section class="sec">
  <div class="sechead"><span class="num">§4</span><h2>Closures on the Fisher-selected knobs</h2></div>
  <div class="reading">
  <p><b>What.</b> Inject a known θ* into the model, generate Asimov "data", fit it back with the
  Fisher-selected knobs; truth must return within errors.</p>
  <p><b>How.</b> <code>physical_fit_run.py</code>, methods M0 (Levenberg–Marquardt χ²+prior), M1 (Gate II:
  per-knob Cochran's Q + vector split-fit → freeze / flag / excise-refit), M2 (Huber IRLS). The fit subset
  is read <em>blind</em> from the Gate-I npz (<code>shrink &lt; 0.5</code>). Binned data/σ/model curves are
  persisted, so the figure re-renders with no bank pass.</p>
  <p class="deffile mono">analysis/paper/physfit/physical_fit_run.py · physfit_closure5_fig.py</p>
  </div>
  {figure("figures/physfit_fig7_closure5.png", "<b>5-parameter closure.</b> Injected θ* (M_A_res, kF_sf, Eb_shift, s_NN_elastic[pn], f_NN_cex) recovered to ≤ 0.5σ; M0 and M2 agree. Bands = ADoNIS-MC; model curves as bin-edge steps.")}
</section>

<section class="sec">
  <div class="sechead"><span class="num">§5</span><h2>Fitting data the model can't describe</h2></div>
  <p class="tag">The unknown-unknown taxonomy: inside vs outside the model manifold.</p>
  <div class="reading">
  <p><b>What.</b> Fake data is built with a distortion the model cannot exactly represent, then fitted. All
  mechanisms live in one function <code>physical_fit_run.apply_mode</code>: a per-event Q²-shape
  distortion <code>w(Q²)=1−amp·exp(−Q²/λ)</code> (<em>inside</em> the manifold — a flexible K-knot
  log-Q² nuisance absorbs it); a GENIE / NEUT foreign generator as data; a GENIE MEC-only (2p2h)
  admixture (<em>outside</em> the manifold).</p>
  <p><b>The defense.</b> M1's Gate II: Cochran's Q heterogeneity per knob + a worst-eigen-direction
  split-fit flag pre-fit, then contiguous-residual flagging and <b>region excision → refit</b>. Inside →
  the Q²-nuisance recovers truth; outside → coherence gating + excision localizes the artifact instead of
  biasing the knobs.</p>
  <p class="deffile mono">analysis/paper/physfit/physical_fit_run.py (apply_mode, gate2_Q, gate2_split) · physfit_*_fig.py</p>
  </div>
  {figure("figures/physfit_fig3_inject2x.png", "<b>Outside-manifold artifact (×2 on δp_T&gt;300).</b> M0 (naive) biases −27σ; M1 refuses the pull (split-fit p=3e-12) and localizes the artifact to the excised region.")}
  {figure("figures/physfit_fig4_genie.png", "<b>GENIE-3M as data.</b> Foreign-generator mismatch; M1 excise-refit recovers the on-manifold parameters with the artifact confined.")}
  {figure("figures/physfit_fig9_q2nuis.png", "<b>Inside-manifold Q²-distortion + nuisance.</b> The flexible g(Q²;c) nuisance recovers truth and reconstructs the injected w(Q²) — perfect knob/nuisance separation.")}
  {figure("figures/physfit_fig10_mecmix.png", "<b>2p2h/MEC admixture.</b> M0 corrupted (−23σ); M1 recovers truth and maps the 2p2h kinematic habitat.")}
</section>

<section class="sec">
  <div class="sechead"><h2>Appendix — fit diagnostics: landscape, trajectory, coverage</h2></div>
  <div class="reading">
  <p><b>What.</b> Three stress tests of the §4 closure-fit machinery on the 10M bank, run as SLURM
  campaigns: (i) the LM/Gauss–Newton landscape around the best fit, with the EXACT χ² (one full bank
  reweight per grid point, 36 pair grids) overlaid on the quadratic expansion; (ii) a 479-toy coverage
  ensemble — each toy throws truth from the ~20% knob priors (Eb_shift from its half-normal: the knob is
  one-sided, the ACHILLES-faithful clamped SF makes Eb&lt;0 non-smooth) and fluctuates the Asimov data by
  its per-bin σ (stat ⊕ 5% syst), then refits; (iii) the postfit χ² and pull distributions of that
  ensemble against their asymptotic expectations.</p>
  <p><b>Findings.</b> (1) The landscape is quadratic to RMS(Δχ²) ≤ 0.4 inside 3σ for all pairs except
  the two thin degeneracy valleys — M_A_res×res_axial_strength (2.3) and s_NN_elastic[1]×f_NN_cex (1.5)
  — and the one-sided Eb_shift row (~0.6). (2) Postfit χ² follows χ²(n_bins) over the converged 95.8%; pulls (converged toys, N=459,
  σ error ±0.03) are unit-normal for 6/9 knobs (0.97–1.06). The two knobs of the most non-quadratic
  valley — M_A_res (σ_pull 1.59) and res_axial_strength (1.69) — have Hessian errors underestimated by
  ~60%: use profile-likelihood errors for that pair. Eb_shift keeps its boundary asymmetry
  (μ=−0.9, σ=2.8). (3) 4.2% of toys fail to converge (χ²
  ≳ 260) — all with Eb truths far up the half-normal tail (median Eb* 6.6 MeV vs 2.7 for converged):
  the fit stalls in an FSI-compensation valley. Short damped steps (PHYSFIT_STEP_SCALE=0.5) plus an
  Eb≥ε box constraint (below zero the clamped model is exactly flat — an absorbing trap) are the
  robust optimizer settings, adopted for the §4/§5 fits; no multi-start anywhere.</p>
  <p class="deffile mono">analysis/paper/physfit/physfit_traj.py · physfit_traj_fig.py · physfit_coverage.py ·
  physfit_coverage_fig.py</p>
  </div>
  {figure("figures/physfit_traj_closure5_params_vs_iter.png", "<b>Fit trajectory.</b> All 9 Gate-I knobs vs LM/GN iteration (short steps, STEP_SCALE=0.55): smooth convergence onto the injected truth within the final ±1σ bands; χ² 2585 → 2.3 in ~10 iterations.")}
  {figure("figures/physfit_traj_closure5_corner.png", "<b>Landscape corner.</b> GN quadratic Δχ² bands (1/2/3σ), −∇χ² streamlines, the actual fit path (blue), and the EXACT Δχ² contours (dashed red; 25–31² full-model grids per pair). The exact contours confirm the quadratic everywhere except the two thin valleys and the one-sided Eb direction.")}
  {figure("figures/physfit_traj_closure5_quadfidelity.png", "<b>Quadratic fidelity.</b> RMS(Δχ²_exact − Δχ²_quad) inside the 3σ region per knob pair (1σ = 2.3 on this scale).")}
  {figure("figures/coverage_chi2_total.png", "<b>Coverage: postfit χ².</b> Total (data+prior) χ² over the toy ensemble vs χ²(n_bins=188); KS p ≈ 0.26. Tail entries are large-Eb-truth toys (see text).")}
  {figure("figures/coverage_pulls.png", "<b>Coverage: pulls.</b> (θ_fit−θ*)/σ_fit per knob vs N(0,1). Unit-normal for 6/9 knobs; the M_A_res/res_axial valley pair runs ~60% wide (profile-likelihood errors recommended); Eb_shift shows its boundary asymmetry.")}
</section>

<section class="sec">
  <div class="sechead"><h2>Open items — readiness gaps</h2></div>
  <ul>
  <li><b>Knob × sample figure (§3).</b> The beam Fisher result is a table + <code>beam_fisher.npz</code>;
  it needs a knob×sample shrinkage heatmap to become a paper figure.</li>
  <li><b>Soft-π⁺ σ NaN (parked, decision pending).</b> Oset σ_abs is NaN for E&lt;m_π because the RES
  generator builds the pion with the m_π0 kinematic mass while the cascade uses the physical mass; the
  walk silently drops ~1.3% of pion interactions. <b>ACHILLES has the identical pathology</b>, so the
  forward walk is faithful — a fix (put the cascade pion on-shell w.r.t. its own mass) is a deliberate
  divergence. <code>docs/logbook/pion_soft_sigma_nan.md</code>.</li>
  <li><b>e–C driver (§3 extension, not built).</b> The faithful EM QE current is implemented + gate-passed
  (<code>dirac.py</code> probe="EM": purely vector, per-nucleon FFs, CC path bit-unchanged). The driver —
  sampling both nucleon species, EM normalization, inclusive dσ/dω, ACHILLES-oracle validation — remains.
  No e–C row enters the Fisher until it passes a validation gate. <code>docs/logbook/electron_scattering.md</code>.</li>
  <li><b>CC0π selection (§1).</b> Figure uses topological; Gate I uses primary-pion-absorbed. ~1–3%
  difference; stated, not reconciled.</li>
  <li><b>§4–5 write-up.</b> Figures exist; the paper narrative for the taxonomy and closures is not
  written.</li>
  </ul>
</section>

</div>
"""


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    html = f"<style>{CSS}</style>\n{BODY}"
    DEST.write_text(html)
    print(f"[out] {DEST}  ({DEST.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
