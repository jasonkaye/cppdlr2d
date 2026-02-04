/*!
 * \file coefs2eval_if_test.cpp
 * \brief Unit tests for coefs2eval_if function (single and multiple points)
 */

#include "../../programs/hubatom/hubatom.hpp"
#include <fmt/format.h>
#include <gtest/gtest.h>

using namespace cppdlr2d;

/*!
 * \brief Test that coefs2eval_if with multiple points matches single-point
 * version
 *
 * This test sets up 2D DLR expansions for chi_s (singlet correlator, pp
 * channel) and chi_m (magnetic correlator, ph channel) of the Hubbard atom,
 * then evaluates them at a list of test points using both the single-point and
 * multiple-point versions of coefs2eval_if, checking that they match.
 */
TEST(Coefs2EvalIfTest, SingleVsMultiplePoints) {

  double beta   = 5.0;
  double u      = 2.0;
  double lambda = 10.0;
  double eps    = 1e-8;
  double tol    = 1e-14;

  // Build 2D DLR grid
  auto [dlr2d_if, dlr2d_rf] = build_dlr2d(lambda, eps, true, false);
  auto dlr2d_if_ph          = get_dlr2d_if_ph(dlr2d_if);

  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  int r       = dlr_rf.size();

  // Build kernel matrix
  auto cf2if = build_cf2if(beta, dlr_rf, dlr2d_if, dlr2d_rf);
  int niom   = dlr2d_if.shape(0);

  // Evaluate chi_s (pp channel) and chi_m (ph channel) on 2D DLR grid
  auto chi_s               = nda::vector<dcomplex>(niom);
  auto chi_m               = nda::vector<dcomplex>(niom);
  std::complex<double> nu1 = 0, nu2 = 0;
  for (int k = 0; k < niom; ++k) {
    // Particle-particle channel
    nu1      = (2 * dlr2d_if(k, 0) + 1) * pi * 1i / beta;
    nu2      = (2 * dlr2d_if(k, 1) + 1) * pi * 1i / beta;
    chi_s(k) = chi_s_fun(u, beta, nu1, nu2);

    // Particle-hole channel
    nu1      = (2 * dlr2d_if_ph(k, 0) + 1) * pi * 1i / beta;
    nu2      = (2 * dlr2d_if_ph(k, 1) + 1) * pi * 1i / beta;
    chi_m(k) = chi_m_fun(u, beta, nu1, nu2);
  }

  // Obtain DLR coefficients
  auto [chi_s_c, chi_s_csing] = vals2coefs(r, cf2if, chi_s, dlr2d_rf);
  auto [chi_m_c, chi_m_csing] = vals2coefs(r, cf2if, chi_m, dlr2d_rf);

  // Define test points
  auto m   = nda::vector<int>({-5, -3, -1, 0, 1, 2, 4, 7});
  auto n   = nda::vector<int>({-4, -2, 0, 1, 3, 5, 6, 8});
  int npts = m.size();

  // Test particle-particle channel (channel = 1)
  auto result_many_pp = coefs2eval_if(beta, dlr_rf, chi_s_c, chi_s_csing, m, n, 1);
  double err_pp       = 0;
  for (int i = 0; i < npts; ++i) {
    auto result_single = coefs2eval_if(beta, dlr_rf, chi_s_c, chi_s_csing, m(i), n(i), 1);
    err_pp             = std::max(err_pp, std::abs(result_many_pp(i) - result_single));
  }
  EXPECT_LT(err_pp, tol);
  fmt::print("Particle-particle channel max error: {:.3e}\n", err_pp);

  // Test particle-hole channel (channel = 2)
  auto result_many_ph = coefs2eval_if(beta, dlr_rf, chi_m_c, chi_m_csing, m, n, 2);
  double err_ph       = 0;
  for (int i = 0; i < npts; ++i) {
    auto result_single = coefs2eval_if(beta, dlr_rf, chi_m_c, chi_m_csing, m(i), n(i), 2);
    err_ph             = std::max(err_ph, std::abs(result_many_ph(i) - result_single));
  }
  EXPECT_LT(err_ph, tol);
  fmt::print("Particle-hole channel max error: {:.3e}\n", err_ph);
}