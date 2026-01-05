#include "hubatom.hpp"
#include <chrono>
#include <fmt/format.h>
#include <gtest/gtest.h>

/*!
 * \brief Test 2D DLR expansion of singlet correlation function, spin vertex
 * function, and calculation of spin polarization for Hubbard atom
 */
void hubatom_test_driver(double beta, double u, double lambda, double eps,
                         int niomtst, int nbos_tst, bool compressgrid,
                         bool compressbasis, double tol) {

  fmt::print("\nBuilding 2D DLR grid...\n");
  auto start = std::chrono::high_resolution_clock::now();
  // auto dlr2d_if = build_dlr2d_if(lambda, eps);
  auto [dlr2d_if, dlr2d_rf] =
      build_dlr2d(lambda, eps, compressgrid, compressbasis);
  auto end = std::chrono::high_resolution_clock::now();
  fmt::print("\nTime: {}\n\n",
             std::chrono::duration<double>(end - start).count());

  // Get DLR nodes for particle-hole channel
  auto dlr2d_if_ph = get_dlr2d_if_ph(dlr2d_if);

  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  int r = dlr_rf.size(); // # DLR basis functions

  // Build kernel matrix
  auto cf2if = build_cf2if(beta, dlr_rf, dlr2d_if, dlr2d_rf);
  fmt::print("System matrix size = {} x {}\n\n", cf2if.shape(0), cf2if.shape(1));

  int niom = dlr2d_if.shape(0);

  // Get fermionic and bosonic DLR grids
  auto ifops_fer = imfreq_ops(lambda, dlr_rf, Fermion);
  auto ifops_bos = imfreq_ops(lambda, dlr_rf, Boson);
  auto dlr_if_fer = ifops_fer.get_ifnodes();
  auto dlr_if_bos = ifops_bos.get_ifnodes();

  // Evaluate Green's function on 1D DLR grid and obtain its DLR coefficients
  std::complex<double> nu1 = 0, nu2 = 0;
  auto g = nda::vector<dcomplex>(r);
  auto gr = nda::vector<dcomplex>(r); // G reversed: G(-i nu_n)
  for (int k = 0; k < r; ++k) {
    nu1 = (2 * dlr_if_fer(k) + 1) * pi * 1i / beta;
    g(k) = g_fun(u, nu1);
    gr(k) = g_fun(u, -nu1);
  }
  auto gc = nda::array<dcomplex, 1>(ifops_fer.vals2coefs(beta, g));
  auto grc = nda::array<dcomplex, 1>(ifops_fer.vals2coefs(beta, gr));

  // Evaluate density correlation function and singlet vertex function on 2D DLR
  // grid and obtain DLR coefficients
  auto chi_s = nda::vector<dcomplex>(niom);
  auto lam_m = nda::vector<dcomplex>(niom);
  for (int k = 0; k < niom; ++k) {
    // Particle-particle channel
    nu1 = (2 * dlr2d_if(k, 0) + 1) * pi * 1i / beta;
    nu2 = (2 * dlr2d_if(k, 1) + 1) * pi * 1i / beta;
    chi_s(k) = chi_s_fun(u, beta, nu1, nu2);

    // Particle-hole channel
    nu1 = (2 * dlr2d_if_ph(k, 0) + 1) * pi * 1i / beta;
    nu2 = (2 * dlr2d_if_ph(k, 1) + 1) * pi * 1i / beta;
    lam_m(k) = lam_m_fun(u, beta, nu1, nu2);
  }

  fmt::print("Obtaining DLR coefficients...\n");

  start = std::chrono::high_resolution_clock::now();

  auto valsall = fmatrix(niom, 2);
  valsall(_, 0) = chi_s;
  valsall(_, 1) = lam_m;
  auto [coefsall, coefsingall] = vals2coefs_many(r, cf2if, valsall, dlr2d_rf);
  auto chi_s_c = coefsall(0, _, _, _);
  auto lam_m_c = coefsall(1, _, _, _);
  auto chi_s_csing = coefsingall(0, _);
  auto lam_m_csing = coefsingall(1, _);

  end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n\n",
             std::chrono::duration<double>(end - start).count());

  // Test DLR expansion of vertex function
  fmt::print("Testing DLR expansion of vertex function...\n");

  // Evaluate expansion on test grid and measure error
  auto chi_s_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto chi_s_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_m_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_m_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  int midx = 0, nidx = 0;
  start = std::chrono::high_resolution_clock::now();
  for (int m = -niomtst / 2; m < niomtst / 2; ++m) {
    for (int n = -niomtst / 2; n < niomtst / 2; ++n) {
      nu1 = ((2 * m + 1) * pi * 1i) / beta;
      nu2 = ((2 * n + 1) * pi * 1i) / beta;
      midx = niomtst / 2 + m;
      nidx = niomtst / 2 + n;

      // Evaluate true functions
      chi_s_tru(midx, nidx) = chi_s_fun(u, beta, nu1, nu2);
      lam_m_tru(midx, nidx) = lam_m_fun(u, beta, nu1, nu2);

      // Evaluate DLR expansions
      chi_s_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, chi_s_c, chi_s_csing, m, n, 1);
      lam_m_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, lam_m_c, lam_m_csing, m, n, 2);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n\n",
             std::chrono::duration<double>(end - start).count());

  double chi_s_l2 = sqrt(sum(pow(abs(chi_s_tru), 2))) / beta / beta;
  double chi_s_linf = max_element(abs(chi_s_tru));
  double chi_s_l2err =
      sqrt(sum(pow(abs(chi_s_tru - chi_s_tst), 2))) / beta / beta;
  double chi_s_linferr = max_element(abs(chi_s_tru - chi_s_tst));

  double lam_m_l2 = sqrt(sum(pow(abs(lam_m_tru), 2))) / beta / beta;
  double lam_m_linf = max_element(abs(lam_m_tru));
  double lam_m_l2err =
      sqrt(sum(pow(abs(lam_m_tru - lam_m_tst), 2))) / beta / beta;
  double lam_m_linferr = max_element(abs(lam_m_tru - lam_m_tst));

  fmt::print("--- chi_s results ---\n");
  fmt::print("L2 norm:    {}\n", chi_s_l2);
  fmt::print("Linf norm:  {}\n", chi_s_linf);
  fmt::print("L2 error:   {}\n", chi_s_l2err);
  fmt::print("Linf error: {}\n\n", chi_s_linferr);

  fmt::print("--- lambda_M results ---\n");
  fmt::print("L2 norm:    {}\n", lam_m_l2);
  fmt::print("Linf norm:  {}\n", lam_m_linf);
  fmt::print("L2 error:   {}\n", lam_m_l2err);
  fmt::print("Linf error: {}\n\n", lam_m_linferr);

  EXPECT_LT(chi_s_l2err, tol);
  EXPECT_LT(lam_m_l2err, tol);

  // Compute polarization from DLR expansions
  auto itops = imtime_ops(lambda, dlr_rf);

  auto pol_m = polarization(beta, lambda, eps, itops, ifops_fer, ifops_bos, grc,
                            gc, lam_m_c, lam_m_csing);
  auto pol_m_c = ifops_bos.vals2coefs(beta, pol_m); // DLR expansion

  // Compute true polarization
  std::complex<double> pol0_m_tru = beta * -k_it(0.0, u / 2, beta) /
                                    (-beta * u * -k_it(0.0, u / 2, beta) - 2);

  // Evaluate polarization on dense grid
  auto pol_m_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_m_tru = nda::vector<dcomplex>(nbos_tst);
  for (int n = -nbos_tst / 2; n < nbos_tst / 2; ++n) {
    pol_m_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_m_c, n);

    if (n == 0) {
      pol_m_tru(n + nbos_tst / 2) = pol0_m_tru;
    } else {
      pol_m_tru(n + nbos_tst / 2) = 0;
    }
  }

  double pol_m_l2 = sqrt(sum(pow(abs(pol_m_tru), 2))) / beta;
  double pol_m_linf = max_element(abs(pol_m_tru));
  double pol_m_l2err = sqrt(sum(pow(abs(pol_m_tru - pol_m_tst), 2))) / beta;
  double pol_m_linferr = max_element(abs(pol_m_tru - pol_m_tst));

  fmt::print("--- pol_m results ---\n");
  fmt::print("L2 norm:    {}\n", pol_m_l2);
  fmt::print("Linf norm:  {}\n", pol_m_linf);
  fmt::print("L2 error:   {}\n", pol_m_l2err);
  fmt::print("Linf error: {}\n\n", pol_m_linferr);

  EXPECT_LT(pol_m_l2err, tol);
}

TEST(hubatom, compress_if) {
  double beta = 64;           // Inverse temperature
  double u = 1.0;             // Interaction
  double lambda = 64;         // DLR cutoff
  double eps = 1e-12;         // DLR tolerance
  int niomtst = 512;          // # imag freq test points (must be even)
  int nbos_tst = 64;          // # pts in test grid for polarization
  bool compressgrid = true;   // Compress grid using pivoted QR?
  bool compressbasis = false; // Compress basis using pivoted QR?
  double tol = 10 * eps;      // Test tolerance
  hubatom_test_driver(beta, u, lambda, eps, niomtst, nbos_tst, compressgrid,
                      compressbasis, tol);
}

TEST(hubatom, compress_rf) {
  double beta = 64;          // Inverse temperature
  double u = 1.0;            // Interaction
  double lambda = 64;        // DLR cutoff
  double eps = 1e-12;        // DLR tolerance
  int niomtst = 512;         // # imag freq test points (must be even)
  int nbos_tst = 64;         // # pts in test grid for polarization
  bool compressgrid = false; // Compress grid using pivoted QR?
  bool compressbasis = true; // Compress basis using pivoted QR?
  double tol = 10 * eps;     // Test tolerance
  hubatom_test_driver(beta, u, lambda, eps, niomtst, nbos_tst, compressgrid,
                      compressbasis, tol);
}

TEST(hubatom, compress_if_rf) {
  double beta = 64;          // Inverse temperature
  double u = 1.0;            // Interaction
  double lambda = 64;        // DLR cutoff
  double eps = 1e-12;        // DLR tolerance
  int niomtst = 512;         // # imag freq test points (must be even)
  int nbos_tst = 64;         // # pts in test grid for polarization
  bool compressgrid = true;  // Compress grid using pivoted QR?
  bool compressbasis = true; // Compress basis using pivoted QR?
  double tol = 100 * eps;    // Test tolerance
  hubatom_test_driver(beta, u, lambda, eps, niomtst, nbos_tst, compressgrid,
                      compressbasis, tol);
}

TEST(hubatom, overcomplete) {
  double beta = 64;           // Inverse temperature
  double u = 1.0;             // Interaction
  double lambda = 64;         // DLR cutoff
  double eps = 1e-12;         // DLR tolerance
  int niomtst = 512;          // # imag freq test points (must be even)
  int nbos_tst = 64;          // # pts in test grid for polarization
  bool compressgrid = false;  // Compress grid using pivoted QR?
  bool compressbasis = false; // Compress basis using pivoted QR?
  double tol = 100 * eps;     // Test tolerance
  hubatom_test_driver(beta, u, lambda, eps, niomtst, nbos_tst, compressgrid,
                      compressbasis, tol);
}