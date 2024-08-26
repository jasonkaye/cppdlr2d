#include "../src/dlr2d.hpp"
#include "../src/utils.hpp"
#include <fmt/format.h>

using namespace dlr2d;

/*!
 * \brief Generate 2D Matsubara frequency DLR grids for several choices of DLR
 * cutoff
 *
 * This is a driver program for the DLR 2D grid generation functions. The user
 * specifies the DLR tolerance eps and a list of DLR cutoff parameters lambda.
 * HDF5 files are then generated in the specified path containing the
 * 2D DLR Matsubara frequency grid points in terms of Matsubara frequency index
 * pairs for the given choices of eps and lambda.
 *
 * We use the method proposed in Kiese et al., "Discrete Lehmann representation
 * of three-point functions", arXiv:2405.06716, which involves building the fine
 * Matsubara frequency grid from combinations of 1D DLR grid points. If
 * threeterm is set to true, it uses a three-term version of the Lehmann
 * representation, obtained by absorbing one of the terms into the others,
 * rather than the one presented in that paper.
 */
int main() {

  double eps = 1e-12;                 // DLR tolerance
  bool threeterm = false;             // 2+1 or 3+1-term 2D DLR
  bool compressbasis = true;          // Overcomplete or compressed basis
  auto path = "../../dlr2d_if_data/"; // Path for DLR 2D grid data

  // auto lambdas = nda::vector<double>(
  //     {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0});
  // auto lambdas =
  //     nda::vector<double>({1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0});
  auto lambdas = nda::vector<double>({64.0});

  for (int i = 0; i < lambdas.size(); i++) {

    fmt::print("Obtaining 2D imag freq DLR grid...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (threeterm) {
      auto filename = get_filename_3term(lambdas(i), eps);
      build_dlr2d_if_3term(lambdas(i), eps, path, filename);
    } else if (compressbasis) {
      auto filename = get_filename(lambdas(i), eps, true);
      build_dlr2d_ifrf(lambdas(i), eps, path, filename);
    } else {
      auto filename = get_filename(lambdas(i), eps);
      build_dlr2d_if(lambdas(i), eps, path, filename);
    }
    auto end = std::chrono::high_resolution_clock::now();
    fmt::print("Time: {}\n",
               std::chrono::duration<double>(end - start).count());
  }
}