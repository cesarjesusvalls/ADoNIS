"""Exact ACHILLES constants, taken verbatim from the generator so the JAX fold can
reproduce its single-pion prediction bit-for-bit (see include/Achilles/Constants.hh
and src/Achilles/fortran/currents_pi_dcc.f90).

These differ from the round numbers used in the (older, diagonal) dcc_xsec by <1 MeV,
but they fix the absolute energy/threshold bookkeeping that matters for an exact
reproduction of dsigma/dW.
"""
# include/Achilles/Constants.hh
M_P = 938.27208816       # proton mass  [MeV]
M_N = 939.56542054       # neutron mass [MeV]
M_NUC = (M_P + M_N) / 2.0          # = 938.919 ; mqe = 0.5*(mp+mn) (main_xsec_new.f90:90)
MQE = M_NUC                        # struck-nucleon energy: E_struck = mqe - E_removal
M_PI0 = 134.9764
M_PIP = 139.57018
M_PI = (2 * M_PIP + M_PI0) / 3.0   # = 138.039 ; DCC fpio=(2 f_pi+ + f_pi0)/3 (pion_init)

# currents_pi_dcc.f90: hard kinematic cuts in hadr_curr_matrix_el -> current = 0 outside
W_THR = 1076.957         # piN threshold cut [MeV] (= M_NUC + M_PI to 3 dp)
W_MAX = 2000.0           # upper W cut [MeV]
Q2_MAX = 5.0e6           # upper Q^2 cut [MeV^2]  (Q2 in [0, 5 GeV^2])
