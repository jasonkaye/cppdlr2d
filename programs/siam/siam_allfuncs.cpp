#include "siam.hpp"

using namespace dlr2d;

int main() {
  double beta = 20; // Inverse temperature
  double u = 5;
  double lambda = 300;    // DLR cutoff
  double eps = 1e-12;     // DLR tolerance
  int niom_dense = 100;   // # imag freq sample pts for fine grid (must be even)
  int niomtst = 256;      // # imag freq test points (must be even)
  int nbos_tst = 1024;    // # pts in test grid for polarization
  bool reduced = true;    // Full or reduced fine grid
  bool threeterm = true; // 2+1 or 3+1-term 2D DLR

  if (threeterm) {
    siam_allfuncs_3term(beta, u, lambda, eps, niomtst, nbos_tst);
  } else {
    siam_allfuncs(beta, u, lambda, eps, niomtst, nbos_tst, reduced, niom_dense);
  }
}