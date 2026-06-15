# ADoNIS workflow API (YAML-config generation + analysis)

Standardized YAML configs + thin drivers centralize the cc0pi/cc1pi work and replace ~25 copy-pasted
chi2-loop scripts.  Built 2026-06-15, behavior-preserving (configs reproduce the figures bit-for-bit).

## Use
- Generate: `python -u scripts/adonis_generate.py configs/gen_cc1pi.yaml`  (or gen_cc0pi)
- Analyze:  `python -u scripts/adonis_analyze.py configs/ana_cc1pi.yaml`   (or ana_cc0pi)
- Python:   `from adonis.workflow import load_gen_config, run_generation, load_analysis_config, run_analysis`

## Modules (adonis/workflow/)
- `materials.py` — chemical-formula -> NuclearTarget registry.  ONLY C (cascade: c12_density + QMC A=12
  + pke12{n,p}) and free-H (primary-level) are wired; anything else raises UnsupportedMaterial (no
  silent carbon fallback).  Adding a nucleus = add its inputs + a NuclearTarget.
- `config.py` — GenConfig / AnalysisConfig / SignalDef / ObservableSpec / CascadeHyperparams /
  TrackingConfig / DataOverlay + YAML loaders.  `cos70` and `pi` sentinels resolve to exact runtime
  floats so cuts/edges match the scripts bit-for-bit; unknown YAML keys are an error.
- `observables.py` — numpy-batch single source: tki (=cc1pi_fig_tki.observables), cc0pi_obs
  (=cc0pi_engine_combined._obs), vertex_W_Q2.  TKI masses imported from kinematics (M_A_12C/M_A_11B).
- `signal.py` — `select_signal(bank, sd)` on the RICH ENGINE schema (pion_id pip|anypi|none,
  proton_lead in_window|global) + `select_reference` (reuses cc1pi_signal.ach_select for CC1pi, mirrors
  the CC0pi rich selection).  ONE function reproduces engine_signal / qe_created_signal / _select_engine.
- `plotting.py` — `chi2_ratio_panel` + `make_figure`: the ONE chi2+ACH/ADO-ratio loop.
- `analyze.py` — config-driven channel composition (CC1pi = res + qe-created banks; CC0pi = qe +
  res-absorbed banks, both via select_signal) + reference + figure.
- `generate.py` — gen_cc_engine_rich logic, importable; bit-IDENTICAL to the script.  `_N_RECOIL` is set
  by the driver (ADONIS_N_RECOIL) before the cascade import.  Tracking summary (TK.finalize over nterms)
  -> `*_truth.npz` sidecar when tracking.enabled.
- `data_overlay.py` — npz (t2k_cc0pi_stv_data) + nuisance-txt loaders.

## Behavior-preserving verification (the acceptance gate)
- S1 materials: CH->{C,H}, Ar raises.  S2: config edges == cc{0,1}pi VARS.  S3: observables EXACT vs
  scripts (np.array_equal).  S4: all 6 ADoNIS+reference selections EXACT.  S5: chi2/ndf + ACH/ADO ==
  scripts.  S6: `ana_cc{0,1}pi.yaml` reproduce CC1pi ACH/ADO 1.012 (pn 0.32 ... lp_p 0.59) and CC0pi
  1.011 (dpt 5.04, W 4.70 ...) line-for-line.  S7: workflow generate vs gen_cc_engine_rich -> ALL arrays
  bit-identical (1866 ev); tracking sidecar parent-linkage 0 dangling.  S8: 7 superseded scripts archived
  to scripts/_legacy/ (grep-verified no importers); configs still render; focused pytest green.

## Known limits / next
- Engine generation is carbon-only; free-H is a separate primary bank (`adonis_h` role raises
  NotImplementedError in analyze).  Multi-material compound generation not wired.
- Per-step trajectory storage in batch generation is out of scope (tracking.steps -> error); viz lives
  in scripts/cascade_viz.py.
- `_N_RECOIL` is still env-driven (read at cascade_discrete import); a config->setter refactor is deferred.
- The other ~25 duplicated chi2-loop scripts still to migrate to workflow.plotting (incremental).
- KEEP: cc1pi_engine_plot / cc0pi_engine_combined (parity reference, imported by chi2_validity),
  cc1pi_signal (ach_select) and cc1pi_fig_tki (observables), still imported by adonis/workflow.
