# Oracle: ACHILLES single-pion-production reference

ACHILLES (built at `../Achilles/build/bin/achilles`, see ../INPUTS.md for the build
recipe) is the forward-fidelity oracle for the differentiable DCC assembly.

## Run
```bash
cd ../Achilles
./build/bin/achilles ../diffpi/oracle/res_1pi_12C.yml   # RES single-pion only
```
`res_1pi_12C.yml` isolates the `RES_Spectral_Func` (DCC single-pion) nuclear model
(QE dropped). Output: total + per-channel cross sections (the 4 single-pion charge
channels) and `res_1pi_12C.hepmc` (events for differential distributions).

`res_1pi_12C_reference.txt` records the integrated cross sections for regression.

## Next
Parse the hepmc events into differential distributions (dsigma/dQ^2, dsigma/dW,
pion-momentum spectra) to validate the differentiable DCC cross-section assembly
(diffpi/dcc.py) bin-by-bin against the oracle at nominal knobs.
