#include "hubatom_mc.hpp"
#include "../hubatom/hubatom.hpp"
#include "fmt/base.h"
#include <chrono>
#include <fmt/format.h>
#include <fstream>
#include <tuple>

using namespace cppdlr2d;

void hubatom_mc_compare(double beta, double u, double lambda, double eps,
                        int nmaxtst, bool compressgrid, bool compressbasis,
                        const std::vector<std::string> &datafiles,
                        bool output /* = false */) {
  using arr2d = nda::array<dcomplex, 2>;
  size_t nfiles = datafiles.size();
  std::vector<arr2d> chi_s_data_vec, chi_d_data_vec, chi_m_data_vec;
  std::vector<arr2d> chi_s_tst_dlr_vec, chi_d_tst_dlr_vec, chi_m_tst_dlr_vec;
  std::vector<int> nmax_vec, n_cycles_vec, num_threads_vec;

  // Process each datafile
  for (const auto &datafile : datafiles) {
    auto [chi_s_data, chi_d_data, chi_m_data, chi_s_tst_dlr, chi_d_tst_dlr,
          chi_m_tst_dlr, n_cycles, num_threads] =
        get_dlr_from_ctint_data(beta, lambda, eps, nmaxtst, compressgrid,
                                compressbasis, datafile);
    chi_s_data_vec.push_back(chi_s_data);
    chi_d_data_vec.push_back(chi_d_data);
    chi_m_data_vec.push_back(chi_m_data);
    chi_s_tst_dlr_vec.push_back(chi_s_tst_dlr);
    chi_d_tst_dlr_vec.push_back(chi_d_tst_dlr);
    chi_m_tst_dlr_vec.push_back(chi_m_tst_dlr);
    nmax_vec.push_back(chi_s_data.shape(0) / 2);
    n_cycles_vec.push_back(n_cycles);
    num_threads_vec.push_back(num_threads);
  }

  // Check all nmax are the same
  for (size_t i = 1; i < nmax_vec.size(); ++i) {
    if (nmax_vec[i] != nmax_vec[0]) {
      fmt::print("Error: nmax mismatch between data files ({} vs {})\n",
                 nmax_vec[0], nmax_vec[i]);
      return;
    }
  }
  int nmax = nmax_vec[0];

  // Compute exact reference data on data grid
  auto [chi_s_data_ref, chi_d_data_ref, chi_m_data_ref] =
      get_ref_data(nmax, beta, u);

  // Compute exact reference data on test grid
  auto [chi_s_tst_ref, chi_d_tst_ref, chi_m_tst_ref] =
      get_ref_data(nmaxtst, beta, u);

  // Store error arrays
  std::vector<double> chi_s_data_l2err(nfiles), chi_d_data_l2err(nfiles),
      chi_m_data_l2err(nfiles);
  std::vector<double> chi_s_data_linferr(nfiles), chi_d_data_linferr(nfiles),
      chi_m_data_linferr(nfiles);
  std::vector<double> chi_s_dlr_l2err(nfiles), chi_d_dlr_l2err(nfiles),
      chi_m_dlr_l2err(nfiles);
  std::vector<double> chi_s_dlr_linferr(nfiles), chi_d_dlr_linferr(nfiles),
      chi_m_dlr_linferr(nfiles);

  for (size_t i = 0; i < nfiles; ++i) {
    chi_s_data_l2err[i] = l2_norm(chi_s_data_vec[i] - chi_s_data_ref, beta);
    chi_d_data_l2err[i] = l2_norm(chi_d_data_vec[i] - chi_d_data_ref, beta);
    chi_m_data_l2err[i] = l2_norm(chi_m_data_vec[i] - chi_m_data_ref, beta);
    chi_s_data_linferr[i] = linf_norm(chi_s_data_vec[i] - chi_s_data_ref);
    chi_d_data_linferr[i] = linf_norm(chi_d_data_vec[i] - chi_d_data_ref);
    chi_m_data_linferr[i] = linf_norm(chi_m_data_vec[i] - chi_m_data_ref);

    chi_s_dlr_l2err[i] = l2_norm(chi_s_tst_ref - chi_s_tst_dlr_vec[i], beta);
    chi_d_dlr_l2err[i] = l2_norm(chi_d_tst_ref - chi_d_tst_dlr_vec[i], beta);
    chi_m_dlr_l2err[i] = l2_norm(chi_m_tst_ref - chi_m_tst_dlr_vec[i], beta);
    chi_s_dlr_linferr[i] = linf_norm(chi_s_tst_ref - chi_s_tst_dlr_vec[i]);
    chi_d_dlr_linferr[i] = linf_norm(chi_d_tst_ref - chi_d_tst_dlr_vec[i]);
    chi_m_dlr_linferr[i] = linf_norm(chi_m_tst_ref - chi_m_tst_dlr_vec[i]);
  }

  // Output errors in readable format
  fmt::print("\n===== CT-INT Data vs Reference (data grid) =====\n");
  for (size_t i = 0; i < nfiles; ++i) {
    fmt::print("Run {}: n_cycles = {}, num_threads = {}\n", i, n_cycles_vec[i],
               num_threads_vec[i]);
    fmt::print("  chi_S: l2 error = {:12.5e}, linf error = {:12.5e}\n",
               chi_s_data_l2err[i], chi_s_data_linferr[i]);
    fmt::print("  chi_D: l2 error = {:12.5e}, linf error = {:12.5e}\n",
               chi_d_data_l2err[i], chi_d_data_linferr[i]);
    fmt::print("  chi_M: l2 error = {:12.5e}, linf error = {:12.5e}\n",
               chi_m_data_l2err[i], chi_m_data_linferr[i]);
  }
  fmt::print("\n===== DLR Expansion vs Reference (test grid) =====\n");
  for (size_t i = 0; i < nfiles; ++i) {
    fmt::print("Run {}: n_cycles = {}, num_threads = {}\n", i, n_cycles_vec[i],
               num_threads_vec[i]);
    fmt::print("  chi_S: l2 error = {:12.5e}, linf error = {:12.5e}\n",
               chi_s_dlr_l2err[i], chi_s_dlr_linferr[i]);
    fmt::print("  chi_D: l2 error = {:12.5e}, linf error = {:12.5e}\n",
               chi_d_dlr_l2err[i], chi_d_dlr_linferr[i]);
    fmt::print("  chi_M: l2 error = {:12.5e}, linf error = {:12.5e}\n",
               chi_m_dlr_l2err[i], chi_m_dlr_linferr[i]);
  }

  if (output) {
    h5::file h5out("hubatom_mc_errors.h5", 'w');
    h5_write(h5out, "n_cycles", n_cycles_vec);
    h5_write(h5out, "num_threads", num_threads_vec);
    h5_write(h5out, "chi_s_data_l2err", chi_s_data_l2err);
    h5_write(h5out, "chi_s_data_linferr", chi_s_data_linferr);
    h5_write(h5out, "chi_d_data_l2err", chi_d_data_l2err);
    h5_write(h5out, "chi_d_data_linferr", chi_d_data_linferr);
    h5_write(h5out, "chi_m_data_l2err", chi_m_data_l2err);
    h5_write(h5out, "chi_m_data_linferr", chi_m_data_linferr);
    h5_write(h5out, "chi_s_dlr_l2err", chi_s_dlr_l2err);
    h5_write(h5out, "chi_s_dlr_linferr", chi_s_dlr_linferr);
    h5_write(h5out, "chi_d_dlr_l2err", chi_d_dlr_l2err);
    h5_write(h5out, "chi_d_dlr_linferr", chi_d_dlr_linferr);
    h5_write(h5out, "chi_m_dlr_l2err", chi_m_dlr_l2err);
    h5_write(h5out, "chi_m_dlr_linferr", chi_m_dlr_linferr);
    h5out.close();
  }
}

std::tuple<nda::array<dcomplex, 2>, nda::array<dcomplex, 2>,
           nda::array<dcomplex, 2>>
get_ref_data(int nmax, double beta, double u) {
  auto chi_s_ref = nda::array<dcomplex, 2>(2 * nmax, 2 * nmax);
  auto chi_d_ref = nda::array<dcomplex, 2>(2 * nmax, 2 * nmax);
  auto chi_m_ref = nda::array<dcomplex, 2>(2 * nmax, 2 * nmax);
  std::complex<double> nu1 = 0, nu2 = 0;
  for (int m = -nmax; m < nmax; ++m) {
    for (int n = -nmax; n < nmax; ++n) {
      nu1 = ((2 * m + 1) * pi * 1i) / beta;
      nu2 = ((2 * n + 1) * pi * 1i) / beta;
      chi_s_ref(nmax + m, nmax + n) = chi_s_fun(u, beta, nu1, nu2);
      chi_d_ref(nmax + m, nmax + n) = chi_d_fun(u, beta, nu1, nu2);
      chi_m_ref(nmax + m, nmax + n) = chi_m_fun(u, beta, nu1, nu2);
    }
  }
  return {chi_s_ref, chi_d_ref, chi_m_ref};
}

std::tuple<nda::array<dcomplex, 2>, nda::array<dcomplex, 2>,
           nda::array<dcomplex, 2>, int, int>
load_chi_data(const std::string &datafile) {
  h5::file ctintdata(datafile, 'r');
  auto chi_pp_rawdata = std::vector<nda::array<double, 7>>(4);
  auto chi_ph_rawdata = std::vector<nda::array<double, 7>>(4);
  h5_read(ctintdata, "chi3pp_iw/up_up/data", chi_pp_rawdata[0]);
  h5_read(ctintdata, "chi3pp_iw/up_dn/data", chi_pp_rawdata[1]);
  h5_read(ctintdata, "chi3pp_iw/dn_up/data", chi_pp_rawdata[2]);
  h5_read(ctintdata, "chi3pp_iw/dn_dn/data", chi_pp_rawdata[3]);
  h5_read(ctintdata, "chi3ph_iw/up_up/data", chi_ph_rawdata[0]);
  h5_read(ctintdata, "chi3ph_iw/up_dn/data", chi_ph_rawdata[1]);
  h5_read(ctintdata, "chi3ph_iw/dn_up/data", chi_ph_rawdata[2]);
  h5_read(ctintdata, "chi3ph_iw/dn_dn/data", chi_ph_rawdata[3]);
  int nmax = chi_pp_rawdata[0].shape(0) / 2;

  auto chi_pp_data = nda::array<dcomplex, 3>(4, 2 * nmax, 2 * nmax);
  auto chi_ph_data = nda::array<dcomplex, 3>(4, 2 * nmax, 2 * nmax);
  for (int i = 0; i < 4; ++i) {
    chi_pp_data(i, _, _) = chi_pp_rawdata[i](_, _, 0, 0, 0, 0, 0) +
                           1i * chi_pp_rawdata[i](_, _, 0, 0, 0, 0, 1);
    chi_ph_data(i, _, _) = chi_ph_rawdata[i](_, _, 0, 0, 0, 0, 0) +
                           1i * chi_ph_rawdata[i](_, _, 0, 0, 0, 0, 1);
  }

  auto chi_s_data = chi_pp_data(1, _, _) - chi_pp_data(2, _, _);
  auto chi_d_data = chi_ph_data(0, _, _) + chi_ph_data(1, _, _);
  auto chi_m_data = chi_ph_data(0, _, _) - chi_ph_data(1, _, _);

  // Extract n_cycles and num_threads
  long n_cycles = 0, num_threads = 0;
  h5_read(ctintdata, "Solver_Info/solve_params/n_cycles", n_cycles);
  h5_read(ctintdata, "Solver_Info/num_threads", num_threads);

  return {chi_s_data, chi_d_data, chi_m_data, n_cycles, num_threads};
}

std::tuple<nda::array<dcomplex, 2>, nda::array<dcomplex, 2>,
           nda::array<dcomplex, 2>, nda::array<dcomplex, 2>,
           nda::array<dcomplex, 2>, nda::array<dcomplex, 2>, int, int>
get_dlr_from_ctint_data(double beta, double lambda, double eps, int nmaxtst,
                        bool compressgrid, bool compressbasis,
                        const std::string &datafile) {
  auto path = "../../../dlr2d_if_data/";

  // Read 2D DLR grid indices from file
  auto filename = get_filename(lambda, eps, compressgrid, compressbasis);
  auto [dlr2d_if, dlr2d_rf] = read_dlr2d(path, filename);
  int n_if = dlr2d_if.shape(0);

  // Get DLR nodes for particle-hole channel
  auto dlr2d_if_ph = get_dlr2d_if_ph(dlr2d_if);

  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  int r = dlr_rf.size();

  // Build kernel matrix
  auto cf2if = build_cf2if(beta, dlr_rf, dlr2d_if, dlr2d_rf);

  // Load CT-INT data
  auto [chi_s_data, chi_d_data, chi_m_data, n_cycles, num_threads] =
      load_chi_data(datafile);
  int nmax = chi_s_data.shape(0) / 2;

  // Subsample CT-INT data on DLR grid
  auto chi_s_data_dlr = nda::vector<dcomplex>(n_if);
  auto chi_d_data_dlr = nda::vector<dcomplex>(n_if);
  auto chi_m_data_dlr = nda::vector<dcomplex>(n_if);
  for (int k = 0; k < n_if; ++k) {
    chi_s_data_dlr(k) =
        chi_s_data(nmax + dlr2d_if(k, 0), nmax + dlr2d_if(k, 1));
    chi_d_data_dlr(k) =
        chi_d_data(nmax + dlr2d_if_ph(k, 0), nmax + dlr2d_if_ph(k, 1));
    chi_m_data_dlr(k) =
        chi_m_data(nmax + dlr2d_if_ph(k, 0), nmax + dlr2d_if_ph(k, 1));
  }

  // Compute DLR coefficients
  auto valsall = fmatrix(n_if, 3);
  valsall(_, 0) = chi_s_data_dlr;
  valsall(_, 1) = chi_d_data_dlr;
  valsall(_, 2) = chi_m_data_dlr;
  auto [coefsall, coefsingall] = vals2coefs_many(r, cf2if, valsall, dlr2d_rf);
  auto chi_s_c = coefsall(0, _, _, _);
  auto chi_d_c = coefsall(1, _, _, _);
  auto chi_m_c = coefsall(2, _, _, _);
  auto chi_s_csing = coefsingall(0, _);
  auto chi_d_csing = coefsingall(1, _);
  auto chi_m_csing = coefsingall(2, _);

  // Evaluate DLR expansion on test grid
  auto chi_s_tst_dlr = nda::array<dcomplex, 2>(2 * nmaxtst, 2 * nmaxtst);
  auto chi_d_tst_dlr = nda::array<dcomplex, 2>(2 * nmaxtst, 2 * nmaxtst);
  auto chi_m_tst_dlr = nda::array<dcomplex, 2>(2 * nmaxtst, 2 * nmaxtst);
  for (int m = -nmaxtst; m < nmaxtst; ++m) {
    for (int n = -nmaxtst; n < nmaxtst; ++n) {
      chi_s_tst_dlr(nmaxtst + m, nmaxtst + n) =
          coefs2eval_if(beta, dlr_rf, chi_s_c, chi_s_csing, m, n, 1);
      chi_d_tst_dlr(nmaxtst + m, nmaxtst + n) =
          coefs2eval_if(beta, dlr_rf, chi_d_c, chi_d_csing, m, n, 2);
      chi_m_tst_dlr(nmaxtst + m, nmaxtst + n) =
          coefs2eval_if(beta, dlr_rf, chi_m_c, chi_m_csing, m, n, 2);
    }
  }
  return {chi_s_data,    chi_d_data,    chi_m_data, chi_s_tst_dlr,
          chi_d_tst_dlr, chi_m_tst_dlr, n_cycles,   num_threads};
}
