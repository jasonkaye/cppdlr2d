/*!
 * \file coefs2eval_if_test.cpp
 * \brief Unit tests for coefs2eval_if function (single and multiple points)
 */

#include "../../programs/hubatom/hubatom.hpp"
#include <fmt/format.h>
#include <gtest/gtest.h>

using namespace cppdlr2d;

// Fixture class for coefs2eval_if tests
class Coefs2EvalIfTest : public ::testing::Test {
  protected:
  double beta   = 5.0;
  double u      = 2.0;
  double lambda = 10.0;
  double eps    = 1e-8;
  double tol    = 1e-14;

  nda::array<int, 2> dlr2d_if;
  nda::array<int, 2> dlr2d_rf;
  nda::array<int, 2> dlr2d_if_ph;
  nda::vector<double> dlr_rf;
  int r;
  fmatrix cf2if;
  int niom;

  // Single expansion coefficients
  nda::array<dcomplex, 3> chi_s_c;
  nda::array<dcomplex, 1> chi_s_csing;
  nda::array<dcomplex, 3> chi_m_c;
  nda::array<dcomplex, 1> chi_m_csing;

  void SetUp() override {
    // Build 2D DLR grid
    std::tie(dlr2d_if, dlr2d_rf) = build_dlr2d(lambda, eps, true, false);
    dlr2d_if_ph                  = get_dlr2d_if_ph(dlr2d_if);

    // Get DLR frequencies
    dlr_rf = build_dlr_rf(lambda, eps);
    r      = dlr_rf.size();

    // Build kernel matrix
    cf2if = build_cf2if(beta, dlr_rf, dlr2d_if, dlr2d_rf);
    niom  = dlr2d_if.shape(0);

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
    std::tie(chi_s_c, chi_s_csing) = vals2coefs(r, cf2if, chi_s, dlr2d_rf);
    std::tie(chi_m_c, chi_m_csing) = vals2coefs(r, cf2if, chi_m, dlr2d_rf);
  }
};

/*!
 * \brief Test that coefs2eval_if with multiple points matches single-point
 * version
 */
TEST_F(Coefs2EvalIfTest, SingleVsMultiplePoints) {

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
  fmt::print("SingleVsMultiple PP channel max error: {:.3e}\n", err_pp);

  // Test particle-hole channel (channel = 2)
  auto result_many_ph = coefs2eval_if(beta, dlr_rf, chi_m_c, chi_m_csing, m, n, 2);
  double err_ph       = 0;
  for (int i = 0; i < npts; ++i) {
    auto result_single = coefs2eval_if(beta, dlr_rf, chi_m_c, chi_m_csing, m(i), n(i), 2);
    err_ph             = std::max(err_ph, std::abs(result_many_ph(i) - result_single));
  }
  EXPECT_LT(err_ph, tol);
  fmt::print("SingleVsMultiple PH channel max error: {:.3e}\n", err_ph);
}

/*!
 * \brief Test batch evaluation with multiple expansions
 */
TEST_F(Coefs2EvalIfTest, BatchEvaluation) {

  // Define test points
  auto m   = nda::vector<int>({-5, -3, -1, 0, 1, 2, 4, 7});
  auto n   = nda::vector<int>({-4, -2, 0, 1, 3, 5, 6, 8});
  int npts = m.size();

  // Create batch with chi_s and 2*chi_s (both PP channel functions)
  // New shape: (3, r, r, nbatch) and (r, nbatch)
  int nbatch               = 2;
  auto gc_reg_batch        = nda::array<dcomplex, 4>(3, r, r, nbatch);
  auto gc_sng_batch        = nda::array<dcomplex, 2>(r, nbatch);
  gc_reg_batch(_, _, _, 0) = chi_s_c;
  gc_sng_batch(_, 0)       = chi_s_csing;
  gc_reg_batch(_, _, _, 1) = 2.0 * chi_s_c;
  gc_sng_batch(_, 1)       = 2.0 * chi_s_csing;

  // Evaluate batch in particle-particle channel
  auto result_batch = coefs2eval_if_many(beta, dlr_rf, gc_reg_batch, gc_sng_batch, m, n, 1);

  // Verify each batch element matches single evaluation
  auto result_s = coefs2eval_if(beta, dlr_rf, chi_s_c, chi_s_csing, m, n, 1);

  double err0 = 0, err1 = 0;
  for (int i = 0; i < npts; ++i) {
    err0 = std::max(err0, std::abs(result_batch(0, i) - result_s(i)));
    err1 = std::max(err1, std::abs(result_batch(1, i) - 2.0 * result_s(i)));
  }
  EXPECT_LT(err0, tol);
  EXPECT_LT(err1, tol);
  fmt::print("Batch PP channel errors: batch[0]={:.3e}, batch[1]={:.3e}\n", err0, err1);

  // Test particle-hole channel with chi_m and 2*chi_m
  gc_reg_batch(_, _, _, 0) = chi_m_c;
  gc_sng_batch(_, 0)       = chi_m_csing;
  gc_reg_batch(_, _, _, 1) = 2.0 * chi_m_c;
  gc_sng_batch(_, 1)       = 2.0 * chi_m_csing;

  result_batch = coefs2eval_if_many(beta, dlr_rf, gc_reg_batch, gc_sng_batch, m, n, 2);

  auto result_m = coefs2eval_if(beta, dlr_rf, chi_m_c, chi_m_csing, m, n, 2);

  err0 = 0;
  err1 = 0;
  for (int i = 0; i < npts; ++i) {
    err0 = std::max(err0, std::abs(result_batch(0, i) - result_m(i)));
    err1 = std::max(err1, std::abs(result_batch(1, i) - 2.0 * result_m(i)));
  }
  EXPECT_LT(err0, tol);
  EXPECT_LT(err1, tol);
  fmt::print("Batch PH channel errors: batch[0]={:.3e}, batch[1]={:.3e}\n", err0, err1);
}

/*!
 * \brief Test that coefs2eval_if_grid matches coefs2eval_if_many with flattened grid
 */
TEST_F(Coefs2EvalIfTest, GridVsMany) {

  int m_min = -3, m_max = 2;
  int n_min = -2, n_max = 3;
  int nm   = m_max - m_min + 1;
  int nn   = n_max - n_min + 1;
  int npts = nm * nn;

  // Pack single expansion into batch format
  // New shape: (3, r, r, nbatch) and (r, nbatch)
  auto gc_reg_batch        = nda::array<dcomplex, 4>(3, r, r, 1);
  auto gc_sng_batch        = nda::array<dcomplex, 2>(r, 1);
  gc_reg_batch(_, _, _, 0) = chi_s_c;
  gc_sng_batch(_, 0)       = chi_s_csing;

  // Evaluate using grid function
  auto result_grid = coefs2eval_if_grid(beta, dlr_rf, gc_reg_batch, gc_sng_batch, m_min, m_max, n_min, n_max, 1);

  // Build flattened m, n vectors and evaluate with _many
  auto m_vec = nda::vector<int>(npts);
  auto n_vec = nda::vector<int>(npts);
  for (int im = 0; im < nm; ++im) {
    for (int in = 0; in < nn; ++in) {
      m_vec(im * nn + in) = m_min + im;
      n_vec(im * nn + in) = n_min + in;
    }
  }
  auto result_many = coefs2eval_if_many(beta, dlr_rf, gc_reg_batch, gc_sng_batch, m_vec, n_vec, 1);

  // Compare
  double err = 0;
  for (int im = 0; im < nm; ++im) {
    for (int in = 0; in < nn; ++in) { err = std::max(err, std::abs(result_grid(0, im, in) - result_many(0, im * nn + in))); }
  }
  EXPECT_LT(err, tol);
  fmt::print("GridVsMany max error: {:.3e}\n", err);
}
