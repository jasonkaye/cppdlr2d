#include "hubatom_mc.hpp"

using namespace cppdlr2d;

int main() {
  double beta = 10;          // Inverse temperature
  double u = 2.0;            // Interaction
  double lambda = 20;        // DLR cutoff
  double eps = 1e-5;         // DLR tolerance
  int niomtst = 500;         // # imag freq test points (must be even)
  bool compressgrid = false; // Full or reduced fine grid
  bool compressbasis = true; // Overcomplete or compressed basis
  bool output = false;       // Write results to h5 file

  auto datafiles = std::vector<std::string>{
      "../../../hubatom_mc_data/ctint_nfft_ncycles_2pow14_np96.h5"};

  // auto datafiles = std::vector<std::string>{
  //     "../../../hubatom_mc_data/ctint_nfft_ncycles_2pow14_np96.h5",
  //     "../../../hubatom_mc_data/ctint_nfft_ncycles_2pow16_np96.h5",
  //     "../../../hubatom_mc_data/ctint_nfft_ncycles_2pow18_np96.h5",
  //     "../../../hubatom_mc_data/ctint_nfft_ncycles_2pow20_np96.h5"};

  hubatom_mc_compare(beta, u, lambda, eps, niomtst, compressgrid, compressbasis,
                     datafiles, output);
}