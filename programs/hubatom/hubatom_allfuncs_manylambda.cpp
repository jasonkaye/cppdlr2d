#include "hubatom.hpp"
#include <fstream>

int main() {
  double u = 1.0;             // Interaction
  double eps = 1e-12;         // DLR tolerance
  int niomtst = 1024;         // # imag freq test points (must be even)
  int nbos_tst = 1024;        // # pts in test grid for polarization
  bool compressgrid = true;   // Full or reduced fine grid
  bool compressbasis = false; // Overcomplete or compressed basis
  bool threeterm = false;     // 2+1 or 3+1-term 2D DLR

  auto filename = "hubatom_eps12_compress_if_rf";

  int nexp = 11;
  int nresult = 42;
  auto results = nda::array<double, 2>(nresult, nexp);
  double beta = 0, lambda = 0;
  for (int i = 0; i < nexp; ++i) {
    beta = pow(2.0, i);
    lambda = beta;
    if (threeterm) {
      results(nda::range::all, i) =
          hubatom_allfuncs_3term(beta, u, lambda, eps, niomtst, nbos_tst);
    } else {
      results(nda::range::all, i) = hubatom_allfuncs(
          beta, u, lambda, eps, niomtst, nbos_tst, compressgrid, compressbasis);
    }
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