#include "siam.hpp"

#include <fstream>

using namespace dlr2d;

int main() {
  double beta = 20;          // Inverse temperature
  double u = 5;              // Interaction
  double lambda = 300;       // DLR cutoff
  int niomtst = 1024;        // # imag freq test points (must be even)
  int nbos_tst = 1024;       // # pts in test grid for polarization
  bool threeterm = false;    // 2+1 or 3+1-term 2D DLR
  bool reduced = true;       // Full or reduced fine grid
  bool compressbasis = true; // Overcomplete or compressed basis
  int niom_dense = 100; // # imag freq sample pts for fine grid (must be even)

  auto filename = "siam_recompress";

  int nexp = 6;
  int nresult = 42;
  auto results = nda::array<double, 2>(nresult, nexp);
  double eps = 0;
  for (int i = 0; i < nexp; ++i) {
    eps = pow(10, -2.0 * (i + 1));
    if (threeterm) {
      results(nda::range::all, i) =
          siam_allfuncs_3term(beta, u, lambda, eps, niomtst, nbos_tst);
    } else {
      results(nda::range::all, i) =
          siam_allfuncs(beta, u, lambda, eps, niomtst, nbos_tst, reduced,
                        compressbasis, niom_dense);
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
