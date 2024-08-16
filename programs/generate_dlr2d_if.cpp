#include "../src/dlr2d.hpp"
#include "../src/utils.hpp"
#include <cppdlr/cppdlr.hpp>
#include <fmt/format.h>

using namespace cppdlr;
using namespace dlr2d;

/*!
 * \brief Generate 2D Matsubara frequency DLR grids for several choices of DLR
 * cutoff
 *
 * This is a driver function for the DLR 2D grid generation functions.
 * It generates a HDF5 files in the specified path containing the
 * 2D DLR Matsubara frequency grid points in terms of Matsubara frequency index
 * pairs.
 *
 * It uses the method proposed in Kiese et al., "Discrete Lehmann representation
 * of three-point functions", arXiv:2405.06716, which involves building the fine
 * Matsubara frequency grid from combinations of 1D DLR grid points. If
 * threeterm is set to true, it uses a three-term version of the Lehmann
 * representation, obtained by absorbing one of the terms into the others,
 * rather than the one presented in that paper.
 *
 * \param[in] lambdas     List of DLR cutoff parameters
 * \param[in] eps         Error tolerance
 * \param[in] path        Path to directory in which to save 2D DLR Mat. freqs.
 * \param[in] threeterm   Use four-term (false) or three-term (true) Lehmann
 * representation
 */
void generate_dlr2d_if_driver(nda::vector<double> lambdas, double eps,
                              std::string path, bool threeterm = false) {

  double lambda = 0;
  for (int i = 0; i < lambdas.size(); i++) {

    lambda = lambdas(i); // DLR cutoff

    // Get DLR frequencies
    auto dlr_rf = build_dlr_rf(lambda, eps);
    int r = dlr_rf.size(); // # DLR basis functions

    fmt::print("\nDLR cutoff Lambda = {}\n", lambda);
    fmt::print("DLR tolerance epsilon = {}\n", eps);
    fmt::print("# DLR basis functions = {}\n", r);

    // Get fermionic and bosonic DLR grids
    auto ifops_fer = imfreq_ops(lambda, dlr_rf, Fermion);
    auto ifops_bos = imfreq_ops(lambda, dlr_rf, Boson);
    auto dlr_if_fer = ifops_fer.get_ifnodes();
    auto dlr_if_bos = ifops_bos.get_ifnodes();

    fmt::print("Obtaining 2D imag freq DLR grid...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (threeterm) {
      fmt::print("System matrix shape = {} x {}\n", 2 * r * r, 2 * r * r + r);
      auto filename = get_filename_3term(lambda, eps);
      build_dlr2d_if_3term(dlr_rf, dlr_if_fer, dlr_if_bos, eps, path, filename);
    } else {
      fmt::print("System matrix shape = {} x {}\n", 3 * r * r, 3 * r * r + r);
      auto filename = get_filename(lambda, eps);
      build_dlr2d_if(dlr_rf, dlr_if_fer, dlr_if_bos, eps, path, filename);
    }
    auto end = std::chrono::high_resolution_clock::now();
    fmt::print("Time: {}\n",
               std::chrono::duration<double>(end - start).count());
  }
}

int main() {

  double eps = 1e-12;                 // DLR tolerance
  bool threeterm = true;              // 2+1 or 3+1-term 2D DLR
  auto path = "../../dlr2d_if_data/"; // Path for DLR 2D grid data

  // auto lambdas = nda::vector<double>(
  //     {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0});
  auto lambdas =
      nda::vector<double>({1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0});
  // auto lambdas = nda::vector<double>({300.0});

  generate_dlr2d_if_driver(lambdas, eps, path, threeterm);
}