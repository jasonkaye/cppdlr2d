#include "hubatom.hpp"
#include <fstream>

int main() {
  // double beta = 40;     // Inverse temperature
  double u = 1.0;       // Interaction
  // double lambda = 40;   // DLR cutoff
  double eps = 1e-8;   // DLR tolerance
  int niomtst  = 1024;    // # imag freq test points (must be even)
  int nbos_tst = 1024;   // # pts in test grid for polarization
  bool reduced = true;  // Full or reduced fine grid
  bool compressbasis = false;
  int niom_dense = 100; // # imag freq sample pts for fine grid (must be even)

  auto filename = "hubatom_eps8_tst1024_new";

  int nexp = 11;
  int nresult = 42;
  auto results = nda::array<double, 2>(nresult, nexp);
  double beta = 0, lambda = 0;
  for (int i = 0; i < nexp; ++i) {
    beta = pow(2.0,i);
    lambda = beta;
    results(nda::range::all, i) = hubatom_allfuncs(beta, u, lambda, eps, niomtst,
                                nbos_tst, reduced, compressbasis, niom_dense);
  }

  // Output results to file in double precision
  std::ofstream f(filename);
  f.precision(16);
  for (int i = 0; i < nexp; ++i) {
    for (int j = 0; j < nresult; ++j) {
      f << results(j, i) << "\n";
    }
  }

}