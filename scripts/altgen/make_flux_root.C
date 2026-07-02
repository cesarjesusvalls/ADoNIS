// Build a ROOT TH1D flux histogram for gevgen from Achilles T2K_nu.dat.
// Format: 2 header lines, then "idx elo[GeV] ehi[GeV] flux[cm^-2/50MeV]".
// gevgen samples the histogram SHAPE; absolute norm irrelevant.
// Run inside LUCiD: root -l -b -q 'make_flux_root.C("/work/T2K_nu.dat","/work/t2k_flux.root","t2kflux")'
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "TH1D.h"
#include "TFile.h"

void make_flux_root(const char* src="/work/T2K_nu.dat",
                    const char* out="/work/t2k_flux.root",
                    const char* hist="t2kflux") {
  std::ifstream fh(src);
  std::string line;
  std::vector<double> elo, ehi, val;
  while (std::getline(fh, line)) {
    std::istringstream ss(line);
    double idx, a, b, f;
    if (!(ss >> idx >> a >> b >> f)) continue;  // skips header/non-numeric lines
    elo.push_back(a); ehi.push_back(b); val.push_back(f);
  }
  int nb = val.size();
  std::vector<double> edges(nb + 1);
  for (int i = 0; i < nb; ++i) edges[i] = elo[i];
  edges[nb] = ehi[nb - 1];

  double integ = 0.0;
  for (int i = 0; i < nb; ++i) integ += val[i] * (edges[i+1] - edges[i]);
  printf("[make_flux_root] %d bins, E in [%g, %g] GeV, integral=%.4e\n",
         nb, edges[0], edges[nb], integ);

  TH1D* h = new TH1D(hist, "T2K ND280 numu flux;E_nu [GeV];flux [cm^-2/50MeV]",
                     nb, edges.data());
  for (int i = 0; i < nb; ++i) h->SetBinContent(i + 1, val[i]);

  TFile* tf = new TFile(out, "RECREATE");
  h->Write();
  tf->Close();
  printf("[make_flux_root] wrote %s:%s\n", out, hist);
}
