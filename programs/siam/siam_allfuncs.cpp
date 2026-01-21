#include "siam.hpp"

using namespace cppdlr2d;

int main() {
  double beta = 20;           // Inverse temperature
  double u = 5;               // Interaction
  double lambda = 300;        // DLR cutoff
  double eps = 1e-12;         // DLR tolerance
  int niomtst = 256;          // # imag freq test points (must be even)
  int nbos_tst = 1024;        // # pts in test grid for polarization
  bool compressgrid = true;   // Full or reduced fine grid
  bool compressbasis = false; // Overcomplete or compressed basis
  bool threeterm = false;     // 2+1 or 3+1-term 2D DLR

  if (threeterm) {
    siam_allfuncs_3term(beta, u, lambda, eps, niomtst, nbos_tst);
  } else {
    siam_allfuncs(beta, u, lambda, eps, niomtst, nbos_tst, compressgrid,
                  compressbasis);
  }
}