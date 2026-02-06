/*!
 * \file coefs2eval_if.cpp
 * \brief Benchmarks for coefs2eval_if evaluation functions
 */

#include "./bench_common.hpp"
#include "../programs/hubatom/hubatom.hpp"
#include <numbers>

using std::numbers::pi;

// Fixture that builds DLR grids and expansion coefficients
class Coefs2EvalFixture : public benchmark::Fixture {
  public:
  double beta = 5.0, u = 2.0, lambda = 10.0, eps = 1e-8;

  nda::array<int, 2> dlr2d_if, dlr2d_rf;
  nda::vector<double> dlr_rf;
  int r = 0;
  fmatrix cf2if;

  // Single expansion coefficients
  nda::array<dcomplex, 3> chi_s_c;
  nda::array<dcomplex, 1> chi_s_csing;

  // Batched expansion coefficients (3 x r x r x nbatch) and (r x nbatch)
  nda::array<dcomplex, 4> gc_reg_batch;
  nda::array<dcomplex, 2> gc_sng_batch;

  // Evaluation points (flattened grid)
  nda::vector<int> m_vec, n_vec;
  int m_min = -10, m_max = 10, n_min = -10, n_max = 10;

  void SetUp(const benchmark::State &state) override {
    auto nbatch = state.range(0);

    // Build 2D DLR grid
    std::tie(dlr2d_if, dlr2d_rf) = build_dlr2d(lambda, eps, true, false);

    // Get DLR frequencies
    dlr_rf = build_dlr_rf(lambda, eps);
    r      = dlr_rf.size();

    // Build kernel matrix and evaluate on grid
    cf2if    = build_cf2if(beta, dlr_rf, dlr2d_if, dlr2d_rf);
    int niom = dlr2d_if.shape(0);

    auto chi_s = nda::vector<dcomplex>(niom);
    for (int k = 0; k < niom; ++k) {
      auto nu1 = (2 * dlr2d_if(k, 0) + 1) * pi * 1i / beta;
      auto nu2 = (2 * dlr2d_if(k, 1) + 1) * pi * 1i / beta;
      chi_s(k) = chi_s_fun(u, beta, nu1, nu2);
    }

    // Obtain DLR coefficients
    std::tie(chi_s_c, chi_s_csing) = vals2coefs(r, cf2if, chi_s, dlr2d_rf);

    // Prepare batched coefficients
    gc_reg_batch = nda::array<dcomplex, 4>(3, r, r, nbatch);
    gc_sng_batch = nda::array<dcomplex, 2>(r, nbatch);
    for (int j = 0; j < nbatch; ++j) {
      gc_reg_batch(_, _, _, j) = chi_s_c;
      gc_sng_batch(_, j)       = chi_s_csing;
    }

    // Build evaluation point vectors from grid
    int nm = m_max - m_min + 1;
    int nn = n_max - n_min + 1;
    m_vec  = nda::vector<int>(nm * nn);
    n_vec  = nda::vector<int>(nm * nn);
    for (int im = 0; im < nm; ++im)
      for (int in = 0; in < nn; ++in) {
        m_vec(im * nn + in) = m_min + im;
        n_vec(im * nn + in) = n_min + in;
      }
  }
};

// Baseline: loop calling single-point coefs2eval_if
BENCHMARK_DEFINE_F(Coefs2EvalFixture, Single)(benchmark::State &state) {
  int npts = m_vec.size();
  for (auto _ : state) {
    for (int i = 0; i < npts; ++i) {
      benchmark::DoNotOptimize(coefs2eval_if(beta, dlr_rf, chi_s_c, chi_s_csing, m_vec(i), n_vec(i), 1));
    }
  }
}
BENCHMARK_REGISTER_F(Coefs2EvalFixture, Single)->Arg(1)->Unit(benchmark::kMicrosecond); // NOLINT

// Batch evaluation at arbitrary points
BENCHMARK_DEFINE_F(Coefs2EvalFixture, Many)(benchmark::State &state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(coefs2eval_if_many(beta, dlr_rf, gc_reg_batch, gc_sng_batch, m_vec, n_vec, 1));
  }
}
BENCHMARK_REGISTER_F(Coefs2EvalFixture, Many)->Arg(1)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond); // NOLINT

// Grid evaluation
BENCHMARK_DEFINE_F(Coefs2EvalFixture, Grid)(benchmark::State &state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(coefs2eval_if_grid(beta, dlr_rf, gc_reg_batch, gc_sng_batch, m_min, m_max, n_min, n_max, 1));
  }
}
BENCHMARK_REGISTER_F(Coefs2EvalFixture, Grid)->Arg(1)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond); // NOLINT
