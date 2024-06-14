#include <cppdlr/cppdlr.hpp>
#include <fmt/format.h>
#include <fstream>
#include <nda/nda.hpp>
#include <numbers>

using namespace cppdlr;
using namespace std::numbers;

std::complex<double> ker(std::complex<double> nu, double om) {
  return 1.0 / (nu - om);
}

std::complex<double> dlr2d_coefs2eval(nda::vector<double> om,
                                      nda::array<dcomplex, 3> gc,
                                      nda::array<dcomplex, 1> gc_skel,
                                      std::complex<double> nu1,
                                      std::complex<double> nu2) {

  int r = om.size(); // # DLR basis functions

  // Make sure coefficient array is 3xrxr
  if (gc.shape(0) != 3)
    throw std::runtime_error("First dim of coefficient array must be 3.");
  if ((gc.shape(1) != r) || (gc.shape(2) != r))
    throw std::runtime_error("Second and third dims of coefficient array must "
                             "be # DLR basis functions r.");

  // Evaluate DLR expansion
  std::complex<double> g = 0;
  for (int k = 0; k < r; ++k) {
    for (int l = 0; l < r; ++l) {
      g += gc(0, k, l) * ker(nu1, om(k)) * ker(nu2, om(l)) +
           gc(1, k, l) * ker(nu2, om(k)) * ker(nu1 + nu2, om(l)) +
           gc(2, k, l) * ker(nu1, om(k)) * ker(nu1 + nu2, om(l));
    }
  }
  if (abs(nu1 + nu2) == 0) {
    for (int k = 0; k < r; ++k) {
      g += gc_skel(k) * ker(nu1, om(k));
    }
  }

  return g;
}

// Function to convert linear index of nxn column-major array to index pair
// (zero-indexed)
std::tuple<int, int> ind2sub(int idx, int n) {
  if (idx >= n * n)
    throw std::runtime_error("Index out of bounds.");
  int i = idx % n;
  int j = (idx - i) / n;
  return {i, j};
}

std::complex<double> mygfun(std::complex<double> nu1,
                            std::complex<double> nu2) {
  std::complex<double> val = 0;
  // val = 0.7 * ker(nu2, 0.723) * ker(nu1 + nu2, 0.318) +
  //       0.3 * ker(nu1, 0.288) * ker(nu2, -0.882);
  val = 0.3 * ker(nu1, 0.288) * ker(nu2, -0.882);
  if (abs(nu1 + nu2) == 0) {
    val += 1.5 * ker(nu1, 0.288);
  }

  return val;
}

// Test 2D DLR Matsubara frequency fitting by three different sampling methods:
// (1) dense Matsubara frequency grid, (2) product of 1D DLR grids, and (3)
// sample nodes determined by pivoted QR (skeletonization). Here, we test a
// simultaneous fit to the regular and singular parts of the representation.

int main() {

  double beta = 20;      // Inverse temperature
  double lambda = beta; // DLR cutoff
  double eps = 1e-12;   // DLR tolerance

  // # im freq points
  int niom_dense =
      100;           // # imag freq sample points for dense grid (must be even)
  int niomtst = 200; // # imag freq test points (must be even)

  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  int r = dlr_rf.size();   // # DLR basis functions
  auto om = dlr_rf / beta; // Convert to physical units

  fmt::print("\nDLR cutoff Lambda = {}\n", lambda);
  fmt::print("DLR tolerance epsilon = {}\n", eps);
  fmt::print("# DLR basis functions = {}\n", r);

  // Get dense Matsubara frequency sampling grid
  auto nu = nda::vector<dcomplex>(niom_dense);
  for (int n = -niom_dense / 2; n < niom_dense / 2; ++n) {
    nu(n + niom_dense / 2) = ((2 * n + 1) * pi * 1i) / beta;
  }

  // Get 1D DLR imag freq grid
  auto ifops = imfreq_ops(lambda, dlr_rf, Fermion);
  auto nu_dlr = ((2 * ifops.get_ifnodes() + 1) * pi * 1i) / beta;

  // Get system matrix for dense grid fitting
  fmt::print("\nBuilding dense system matrix...\n");
  auto sysmat_dense =
      nda::matrix<dcomplex, F_layout>(niom_dense * niom_dense, 3 * r * r + r);
  std::complex<double> nu1 = 0, nu2 = 0;

  // Regular part
  for (int k = 0; k < r; ++k) {
    for (int l = 0; l < r; ++l) {
      for (int n = 0; n < niom_dense; ++n) {
        for (int m = 0; m < niom_dense; ++m) {
          sysmat_dense(n * niom_dense + m, k * r + l) =
              ker(nu(m), om(k)) * ker(nu(n), om(l));
          sysmat_dense(n * niom_dense + m, r * r + k * r + l) =
              ker(nu(n), om(k)) * ker(nu(m) + nu(n), om(l));
          sysmat_dense(n * niom_dense + m, 2 * r * r + k * r + l) =
              ker(nu(m), om(k)) * ker(nu(m) + nu(n), om(l));
        }
      }
    }
  }

  // Singular part
  for (int k = 0; k < r; ++k) {
    for (int n = 0; n < niom_dense; ++n) {
      for (int m = 0; m < niom_dense; ++m) {
        if (m == -n - 1) {
          sysmat_dense(n * niom_dense + m, 3 * r * r + k) = ker(nu(m), om(k));
        } else {
          sysmat_dense(n * niom_dense + m, 3 * r * r + k) = 0;
        }
      }
    }
  }

  // Pivoted QR to determine sampling nodes
  fmt::print("Pivoted QR to determine skeleton nodes...\n");
  auto [q, nrm, piv] = pivrgs(sysmat_dense, eps);
  int niom_skel = piv.size();

  // Extract skeleton nodes from pivots
  auto nu_skel = nda::array<dcomplex, 2>(niom_skel, 2);
  for (int k = 0; k < niom_skel; ++k) {
    auto [i, j] = ind2sub(piv(k), niom_dense);
    nu_skel(k, 0) = nu(i);
    nu_skel(k, 1) = nu(j);
  }

  // Obtain skeletonized system matrix
  auto sysmat_skel = nda::matrix<dcomplex, F_layout>(niom_skel, 3 * r * r + r);
  for (int i = 0; i < niom_skel; ++i) {
    sysmat_skel(i, _) = sysmat_dense(piv(i), _);
  }

  // Get three-point function on dense grid
  auto g_dense = nda::array<dcomplex, 2, F_layout>(niom_dense, niom_dense);
  for (int n = 0; n < niom_dense; ++n) {
    for (int m = 0; m < niom_dense; ++m) {
      g_dense(m, n) = mygfun(nu(m), nu(n));
    }
  }

  // Get three-point function on skeleton grid
  auto g_skel = nda::vector<dcomplex>(niom_skel);
  for (int k = 0; k < niom_skel; ++k) {
    g_skel(k) = mygfun(nu_skel(k, 0), nu_skel(k, 1));
  }

  // Get DLR coefficients by fitting from dense grid
  auto s = nda::vector<double>(3 * r * r + r); // Singular values (not needed)
  int rank = 0;                                // Rank (not needed)

  fmt::print("Performing dense grid fit...\n");

  auto g_dense_rs = nda::reshape(g_dense, niom_dense * niom_dense);
  nda::lapack::gelss(sysmat_dense, g_dense_rs, s, 0.0, rank);
  auto gc_dense = nda::array<dcomplex, 3>(3, r, r);
  auto gc_sing_dense = nda::array<dcomplex, 1>(r);
  reshape(gc_dense, 3 * r * r) = g_dense_rs(nda::range(3 * r * r));
  gc_sing_dense = g_dense_rs(nda::range(3 * r * r, 3 * r * r + r));

  // Get DLR coefficients by fitting from skeleton grid
  fmt::print("Performing skeleton grid fit...\n");

  // auto gc_skel = nda::array<dcomplex, 3>(3, r, r);
  // auto gc_skel_rs = reshape(gc_skel, 3 * r * r);
  // gc_skel_rs(nda::range(niom_skel)) = g_skel;
  // nda::lapack::gelss(sysmat_skel, gc_skel_rs, s, 0.0, rank);

  auto coefs = nda::array<dcomplex, 1>(3 * r * r + r);
  coefs(nda::range(niom_skel)) = g_skel;
  nda::lapack::gelss(sysmat_skel, coefs, s, 0.0, rank);
  auto gc_skel = nda::array<dcomplex, 3>(3, r, r);
  auto gc_sing_skel = nda::array<dcomplex, 1>(r);
  reshape(gc_skel, 3 * r * r) = coefs(nda::range(3 * r * r));
  gc_sing_skel = coefs(nda::range(3 * r * r, 3 * r * r + r));

  // Test DLR expansions
  fmt::print("Testing DLR expansions...\n");

  // Get dense Matsubara frequency test grid
  auto nutst = nda::vector<dcomplex>(niomtst);
  for (int n = -niomtst / 2; n < niomtst / 2; ++n) {
    nutst(n + niomtst / 2) = ((2 * n + 1) * pi * 1i) / beta;
  }

  // Evaluate expansion on test grid and measure error
  double linfnrm = 0, l2nrm = 0, diff = 0;
  double linferr_dense = 0, l2err_dense = 0, linferr_skel = 0, l2err_skel = 0;
  auto geval1 = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto geval2 = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto gtrue = nda::array<dcomplex, 2>(niomtst, niomtst);
  for (int n = 0; n < niomtst; ++n) {
    for (int m = 0; m < niomtst; ++m) {
      // Evaluate true function
      gtrue(m, n) = mygfun(nutst(m), nutst(n));
      linfnrm = std::max(linfnrm, abs(gtrue(m, n)));
      l2nrm += pow(abs(gtrue(m, n)), 2);

      // Evaluate dense expansion
      geval1(m, n) =
          dlr2d_coefs2eval(om, gc_dense, gc_sing_dense, nutst(m), nutst(n));
      diff = abs(geval1(m, n) - gtrue(m, n));
      linferr_dense = std::max(linferr_dense, diff);
      l2err_dense += diff * diff;

      // Evaluate skeleton expansion
      geval2(m, n) =
          dlr2d_coefs2eval(om, gc_skel, gc_sing_skel, nutst(m), nutst(n));
      diff = abs(geval2(m, n) - gtrue(m, n));
      linferr_skel = std::max(linferr_skel, diff);
      l2err_skel += diff * diff;
    }
  }

  l2nrm = sqrt(l2nrm) / beta / beta;
  l2err_dense = sqrt(l2err_dense) / beta / beta;
  l2err_skel = sqrt(l2err_skel) / beta / beta;

  fmt::print("\nDense system matrix shape = {} x {}\n", sysmat_dense.shape(0),
             sysmat_dense.shape(1));
  fmt::print("DLR rank squared = {}\n", r * r);
  fmt::print("Pivoted QR rank = {}\n", niom_skel);

  fmt::print("\nL2 norm of function = {}\n", l2nrm);
  fmt::print("Linf norm of function = {}\n", linfnrm);

  fmt::print("\nDense sampling nodes, L2 error = {}\n", l2err_dense);
  fmt::print("Skeleton sampling nodes, L2 error = {}\n", l2err_skel);

  fmt::print("\nDense sampling nodes, Linf error = {}\n", linferr_dense);
  fmt::print("Skeleton sampling nodes, Linf error = {}\n", linferr_skel);

  // Output results to file
  fmt::print("\nWriting results to file...\n");
  std::ofstream f1("gtrue");
  f1.precision(16);
  for (int n = 0; n < niomtst; ++n) {
    for (int m = 0; m < niomtst; ++m) {
      f1 << gtrue(m, n).real() << " " << gtrue(m, n).imag() << "\n";
    }
  }
  f1.close();

  std::ofstream f2("geval_dense");
  f2.precision(16);
  for (int n = 0; n < niomtst; ++n) {
    for (int m = 0; m < niomtst; ++m) {
      f2 << geval1(m, n).real() << " " << geval1(m, n).imag() << "\n";
    }
  }
  f2.close();

  std::ofstream f3("geval_skel");
  f3.precision(16);
  for (int n = 0; n < niomtst; ++n) {
    for (int m = 0; m < niomtst; ++m) {
      f3 << geval2(m, n).real() << " " << geval2(m, n).imag() << "\n";
    }
  }
  f3.close();

  std::ofstream f4("nus");
  f4.precision(16);
  for (int n = 0; n < niomtst; ++n) {
    f4 << nutst(n).imag() << "\n";
  }

  // // Output skeleton nodes to file in double precision
  // fmt::print("\nWriting skeleton nodes to file...\n");
  // std::ofstream f("skel_nodes");
  // f.precision(16);
  // for (int k = 0; k < niom_skel; ++k) {
  //   f << nu_skel(k, 0).imag() << " " << nu_skel(k, 1).imag() << "\n";
  // }
  // f.close();
}