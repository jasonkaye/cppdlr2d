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

  hubatom_chi_d(beta, u, lambda, eps, niomtst,
                                  nbos_tst);
}