#include "hubatom.hpp"

int main() {
  double beta = 64;     // Inverse temperature
  double u = 1.0;       // Interaction
  double lambda = 64;   // DLR cutoff
  double eps = 1e-12;   // DLR tolerance
  int niom_dense = 100; // # imag freq sample pts for fine grid (must be even)
  int niomtst = 512;    // # imag freq test points (must be even)
  int nbos_tst = 64;    // # pts in test grid for polarization
  bool reduced = true;  // Full or reduced fine grid
  bool compressbasis = false;
  bool two_terms = true; // 2+1 or 3+1-term 2D DLR

  if (two_terms) {
    hubatom_allfuncs_two_terms(beta, u, lambda, eps, niomtst, nbos_tst);
  } else {
    auto results = hubatom_allfuncs(beta, u, lambda, eps, niomtst, nbos_tst,
                                    reduced, compressbasis, niom_dense);
  }
}