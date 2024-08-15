#include "../src/dlr2d.hpp"
#include "../src/utils.hpp"
#include <cppdlr/cppdlr.hpp>
#include <fmt/format.h>

using namespace cppdlr;
using namespace dlr2d;

void generate_dlr2d_if_driver(nda::vector<double> lambdas, double eps,
                              nda::vector<int> nioms_dense, bool reduced,
                              bool compressbasis, std::string path) {

  if (lambdas.size() != nioms_dense.size())
    throw std::runtime_error("lambdas and nioms_dense must have the same size");

  double lambda = 0;
  int niom_dense = 0;
  for (int i = 0; i < lambdas.size(); i++) {

    lambda = lambdas(i);         // DLR cutoff
    niom_dense = nioms_dense(i); // # imag freq pts for fine grid

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
    if (!reduced) {
      fmt::print("Fine grid system matrix shape = {} x {}\n",
                 niom_dense * niom_dense, 3 * r * r + r);
      auto filename = get_filename(lambda, eps, niom_dense);
      build_dlr2d_if_fullgrid(dlr_rf, niom_dense, eps, path, filename);
    } else {
      fmt::print("System matrix shape = {} x {}\n", 3 * r * r, 3 * r * r + r);
      auto filename = get_filename(lambda, eps, compressbasis);
      if (!compressbasis) {
        build_dlr2d_if(dlr_rf, dlr_if_fer, dlr_if_bos, eps, path, filename);
      } else {
        build_dlr2d_ifrf(dlr_rf, dlr_if_fer, dlr_if_bos, eps, path, filename);
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    fmt::print("Time: {}\n",
               std::chrono::duration<double>(end - start).count());
  }
}

void generate_dlr2d_if_two_terms_driver(nda::vector<double> lambdas, double eps,
                                        std::string path) {

  double lambda = 0;
  int niom_dense = 0;
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
    fmt::print("System matrix shape = {} x {}\n", 2 * r * r, 2 * r * r + r);

    auto filename = get_filename_3term(lambda, eps);
    build_dlr2d_if_3term(dlr_rf, dlr_if_fer, dlr_if_bos, eps, path, filename);

    auto end = std::chrono::high_resolution_clock::now();
    fmt::print("Time: {}\n",
               std::chrono::duration<double>(end - start).count());
  }
}

int main() {

  double eps = 1e-12;  // DLR tolerance
  bool reduced = true; // Full or reduced fine grid
  bool compressbasis = false;
  bool two_terms = true;              // 2+1 or 3+1-term 2D DLR
  auto path = "../../dlr2d_if_data/"; // Path for DLR 2D grid data

  // auto lambdas = nda::vector<double>(
  //     {1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0});
  auto lambdas =
      nda::vector<double>({1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0});
  // auto lambdas = nda::vector<double>({300.0});

  auto nioms_dense = nda::vector<int>(4 * lambdas);

  if (two_terms) {
    generate_dlr2d_if_two_terms_driver(lambdas, eps, path);
  } else {
    generate_dlr2d_if_driver(lambdas, eps, nioms_dense, reduced, compressbasis,
                             path);
  }
}