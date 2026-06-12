# Oracle: ACHILLES reference run products

This directory holds the `.npz` targets/caches the validation scripts compare against:
ACHILLES oracle histograms/events (generated via the docker images, see
`docs/CONTAINER.md`) and extracted experimental releases (T2K/MINERvA STV via
`../nuisance`). Everything here except this README is **gitignored** (`*.npz`) by
design — the files are large run products, regenerable from `scripts/gen_*.py` /
`scripts/extract_*.py` / `scripts/make_oracle_*.py`; CI fetches its oracle from the
`oracle-data` release asset instead (see README.md "Continuous integration").

If a file is expensive to regenerate (multi-hour ACHILLES runs), keep a copy outside
the repo before cleaning.
