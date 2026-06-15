# Legacy scripts (archived, reversible)

These one-off scripts are **superseded by the YAML-config workflow API** (`adonis/workflow/`, driven
by `scripts/adonis_generate.py` / `scripts/adonis_analyze.py` + `configs/*.yaml`).  They are kept here
for reference, not deleted.  Verified no module imports them at archive time.

| archived script | replaced by |
|---|---|
| `cc0pi_compare.py`      | `configs/ana_cc0pi.yaml` + `scripts/adonis_analyze.py` (engine, absolute) |
| `cc0pi_ratios.py`       | `configs/ana_cc0pi.yaml` (per-observable ACH/ADO ratios) |
| `cc0pi_fig_tki.py`      | `configs/ana_cc0pi.yaml` (dpt/dalphat; data overlay via `data:` block) |
| `cc1pi_ratios.py`       | `configs/ana_cc1pi.yaml` |
| `cc1pi_plot.py`         | `configs/ana_cc1pi.yaml` + `scripts/adonis_analyze.py` |
| `gen_t2k_cc0pi_xsec.py` | `configs/gen_cc0pi.yaml` + `scripts/adonis_generate.py` (engine RICH bank) |
| `gen_t2k_cc1pi_xsec.py` | `configs/gen_cc1pi.yaml` + `scripts/adonis_generate.py` |

The canonical engine scripts `cc1pi_engine_plot.py` / `cc0pi_engine_combined.py` are KEPT (they are the
parity reference for the config workflow and are imported by `chi2_validity.py`).  `cc1pi_signal.py`
(`ach_select`) and `cc1pi_fig_tki.py` (`observables`) are KEPT — still imported by `adonis/workflow/`.

Moderate-risk scripts not yet archived (no config reproduces them yet): `cc0pi_disaggregated.py`,
`cc1pi_disaggregated.py`, `gen_t2k_cc0pi_combined.py`.  The other ~25 duplicated chi2-loop scripts
migrate to `adonis/workflow/plotting.py` incrementally in a later pass.

To restore: `git mv scripts/_legacy/<name>.py scripts/<name>.py`.
