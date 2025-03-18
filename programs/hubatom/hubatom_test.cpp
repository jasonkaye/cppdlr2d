#include "hubatom.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

/*!
 * \brief Test DLR expansion of density correlation function, singlet vertex
 * function, and calculation of singlet polarization for Hubbard atom
 *
 * \note The function tests both the particle/particle and particle/hole
 * DLR expansions, and the algorithm for computing the polarization.
 */
TEST(hubatom, main) {
  double beta = 64;   // Inverse temperature
  double u = 1.0;     // Interaction
  double lambda = 64; // DLR cutoff
  double eps = 1e-12; // DLR tolerance
  int niomtst = 512;  // # imag freq test points (must be even)
  int nbos_tst = 64;  // # pts in test grid for polarization

  auto path = "../../../dlr2d_if_data/"; // Path for DLR 2D grid data
  auto filename = get_filename(lambda, eps);
  auto dlr2d_if = read_dlr2d_if(path, filename);

  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  int r = dlr_rf.size(); // # DLR basis functions

  fmt::print("\nDLR cutoff Lambda = {}\n", lambda);
  fmt::print("DLR tolerance epsilon = {}\n", eps);
  fmt::print("# DLR basis functions = {}\n", r);

  // Get DLR nodes for particle-hole channel
  auto dlr2d_if_ph = nda::array<int, 2>(dlr2d_if.shape());
  dlr2d_if_ph(_, 0) = -dlr2d_if(_, 0) - 1;
  dlr2d_if_ph(_, 1) = dlr2d_if(_, 1);

  auto kmat = build_cf2if(beta, dlr_rf, dlr2d_if);
  fmt::print("Fine system matrix shape = {} x {}\n", 3 * r * r + r,
             3 * r * r + r);

  int niom = dlr2d_if.shape(0);

  fmt::print("DLR rank squared = {}\n", r * r);
  fmt::print("System matrix size = {} x {}\n\n", kmat.shape(0), kmat.shape(1));

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
  auto chi_d = nda::vector<dcomplex>(niom);
  auto lam_s = nda::vector<dcomplex>(niom);
  for (int k = 0; k < niom; ++k) {
    // Particle-hole channel
    nu1 = (2 * dlr2d_if_ph(k, 0) + 1) * pi * 1i / beta;
    nu2 = (2 * dlr2d_if_ph(k, 1) + 1) * pi * 1i / beta;
    chi_d(k) = chi_d_fun(u, beta, nu1, nu2);

    // Particle-particle channel
    nu1 = (2 * dlr2d_if(k, 0) + 1) * pi * 1i / beta;
    nu2 = (2 * dlr2d_if(k, 1) + 1) * pi * 1i / beta;
    lam_s(k) = lam_s_fun(u, beta, nu1, nu2);
  }

  fmt::print("Obtaining DLR coefficients...\n");

  auto start = std::chrono::high_resolution_clock::now();

  auto [chi_d_c, chi_d_csing] = vals2coefs_if(kmat, chi_d, r);
  auto [lam_s_c, lam_s_csing] = vals2coefs_if(kmat, lam_s, r);

  auto end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n", std::chrono::duration<double>(end - start).count());

  // Test DLR expansion of vertex function
  fmt::print("Testing DLR expansion of vertex function...\n");

  // Evaluate expansion on test grid and measure error
  auto chi_d_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto chi_d_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_s_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_s_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  int midx = 0, nidx = 0;
  start = std::chrono::high_resolution_clock::now();
  for (int m = -niomtst / 2; m < niomtst / 2; ++m) {
    for (int n = -niomtst / 2; n < niomtst / 2; ++n) {
      nu1 = ((2 * m + 1) * pi * 1i) / beta;
      nu2 = ((2 * n + 1) * pi * 1i) / beta;
      midx = niomtst / 2 + m;
      nidx = niomtst / 2 + n;

      // Evaluate true functions
      chi_d_tru(midx, nidx) = chi_d_fun(u, beta, nu1, nu2);
      lam_s_tru(midx, nidx) = lam_s_fun(u, beta, nu1, nu2);

      // Evaluate DLR expansions
      chi_d_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, chi_d_c, chi_d_csing, m, n, 2);
      lam_s_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, lam_s_c, lam_s_csing, m, n, 1);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n\n",
             std::chrono::duration<double>(end - start).count());

  double chi_d_l2 = sqrt(sum(pow(abs(chi_d_tru), 2))) / beta / beta;
  double chi_d_linf = max_element(abs(chi_d_tru));
  double chi_d_l2err =
      sqrt(sum(pow(abs(chi_d_tru - chi_d_tst), 2))) / beta / beta;
  double chi_d_linferr = max_element(abs(chi_d_tru - chi_d_tst));

  double lam_s_l2 = sqrt(sum(pow(abs(lam_s_tru), 2))) / beta / beta;
  double lam_s_linf = max_element(abs(lam_s_tru));
  double lam_s_l2err =
      sqrt(sum(pow(abs(lam_s_tru - lam_s_tst), 2))) / beta / beta;
  double lam_s_linferr = max_element(abs(lam_s_tru - lam_s_tst));

  fmt::print("--- chi_D results ---\n");
  fmt::print("L2 norm:    {}\n", chi_d_l2);
  fmt::print("Linf norm:  {}\n", chi_d_linf);
  fmt::print("L2 error:   {}\n", chi_d_l2err);
  fmt::print("Linf error: {}\n\n", chi_d_linferr);

  fmt::print("--- lambda_S results ---\n");
  fmt::print("L2 norm:    {}\n", lam_s_l2);
  fmt::print("Linf norm:  {}\n", lam_s_linf);
  fmt::print("L2 error:   {}\n", lam_s_l2err);
  fmt::print("Linf error: {}\n\n", lam_s_linferr);

  EXPECT_LT(chi_d_l2err, 10 * eps);
  EXPECT_LT(lam_s_l2err, 10 * eps);

  // Compute polarization from DLR expansions
  auto itops = imtime_ops(lambda, dlr_rf);

  auto pol_s = polarization(beta, lambda, eps, itops, ifops_fer, ifops_bos,
                                gc, gc, lam_s_c, lam_s_csing);
  pol_s *= -1.0 / 2;

  auto pol_s_c = ifops_bos.vals2coefs(beta, pol_s); // DLR expansion

  // Compute true polarization
  std::complex<double> pol0_s_tru =
      beta * -k_it(0.0, -u / 2, beta) /
      (2 * beta * u * -k_it(0.0, -u / 2, beta) - 4);

  // Evaluate polarization on dense grid
  auto pol_s_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_s_tru = nda::vector<dcomplex>(nbos_tst);
  for (int n = -nbos_tst / 2; n < nbos_tst / 2; ++n) {
    pol_s_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_s_c, n);

    if (n == 0) {
      pol_s_tru(n + nbos_tst / 2) = pol0_s_tru;
    } else {
      pol_s_tru(n + nbos_tst / 2) = 0;
    }
  }

  double pol_s_l2 = sqrt(sum(pow(abs(pol_s_tru), 2))) / beta;
  double pol_s_linf = max_element(abs(pol_s_tru));
  double pol_s_l2err = sqrt(sum(pow(abs(pol_s_tru - pol_s_tst), 2))) / beta;
  double pol_s_linferr = max_element(abs(pol_s_tru - pol_s_tst));

  fmt::print("--- pol_s results ---\n");
  fmt::print("L2 norm:    {}\n", pol_s_l2);
  fmt::print("Linf norm:  {}\n", pol_s_linf);
  fmt::print("L2 error:   {}\n", pol_s_l2err);
  fmt::print("Linf error: {}\n\n", pol_s_linferr);

  EXPECT_LT(pol_s_l2err, 10 * eps);
}
