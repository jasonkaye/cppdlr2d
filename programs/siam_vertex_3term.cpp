#include "../src/dlr2d.hpp"
#include "../src/utils.hpp"
#include "../src/polarization.hpp"
#include "cppdlr/dlr_kernels.hpp"
#include "nda/blas/tools.hpp"
#include <cppdlr/cppdlr.hpp>
#include <fmt/format.h>
#include <fstream>
#include <nda/nda.hpp>
#include <numbers>

using namespace cppdlr;
using namespace std::numbers;
using namespace dlr2d;

void siam_driver_3term(double beta, double u, double lambda,
                                double eps, int niom_dense, int niomtst,
                                int nbos_tst, bool reduced) {

  auto path = "../../dlr2d_if_data/"; // Path for DLR 2D grid data
  auto datafile =
      "../../siam_data/SIAM_beta20_U5.0_ED_extracted_final.h5"; // Filename for
                                                                // Green's
                                                                // function,
                                                                // correlator,
                                                                // and vertex
                                                                // data
  auto filename = get_filename_3term(lambda, eps);
  auto dlr2d_if = read_dlr2d_if(path, filename);

  // Get DLR nodes for particle-hole channel
  auto dlr2d_if_ph = nda::array<int, 2>(dlr2d_if.shape());
  dlr2d_if_ph(_, 0) = -dlr2d_if(_, 0) - 1;
  dlr2d_if_ph(_, 1) = dlr2d_if(_, 1);

  // Get DLR frequencies
  auto dlr_rf = build_dlr_rf(lambda, eps);
  int r = dlr_rf.size(); // # DLR basis functions

  fmt::print("\nDLR cutoff Lambda = {}\n", lambda);
  fmt::print("DLR tolerance epsilon = {}\n", eps);
  fmt::print("# DLR basis functions = {}\n", r);
  fmt::print("# imag freq in fine grid = {}\n\n", niom_dense);

  // Build kernel matrix
  auto kmat = dlr2d::build_cf2if_3term(beta, dlr_rf, dlr2d_if);
  int niom = dlr2d_if.shape(0);

  fmt::print("Fine system matrix shape = {} x {}\n", 2 * r * r,
            2 * r * r + r);
  fmt::print("DLR rank squared = {}\n", r * r);
  fmt::print("System matrix size = {} x {}\n\n", kmat.shape(0), kmat.shape(1));

  // Get fermionic and bosonic DLR grids
  auto ifops_fer = imfreq_ops(lambda, dlr_rf, Fermion);
  auto ifops_bos = imfreq_ops(lambda, dlr_rf, Boson);
  auto dlr_if_fer = ifops_fer.get_ifnodes();
  auto dlr_if_bos = ifops_bos.get_ifnodes();

  // Load Green's function, correlator, and vertex data
  h5::file siamdata(datafile, 'r');

  auto g_data = nda::array<dcomplex, 1>();
  h5_read(siamdata, "G", g_data);
  int g_nmax = g_data.size() / 2; // Imag freq grid on n = -g_nmax:g_nmax-1

  auto eta_s_data = nda::array<dcomplex, 1>();
  auto eta_d_data = nda::array<dcomplex, 1>();
  auto eta_m_data = nda::array<dcomplex, 1>();
  h5_read(siamdata, "eta_S", eta_s_data);
  h5_read(siamdata, "eta_D", eta_d_data);
  h5_read(siamdata, "eta_M", eta_m_data);
  int eta_nmax =
      (eta_s_data.size() - 1) / 2; // Imag freq grid on n = -eta_nmax:eta_nmax

  auto chi_s_data = nda::array<dcomplex, 2>();
  auto chi_d_data = nda::array<dcomplex, 2>();
  auto chi_m_data = nda::array<dcomplex, 2>();
  h5_read(siamdata, "chi3_S", chi_s_data);
  h5_read(siamdata, "chi3_D", chi_d_data);
  h5_read(siamdata, "chi3_M", chi_m_data);
  int chi_nmax =
      chi_s_data.shape(0) / 2; // Imag freq grid on n = -chi_nmax:chi_nmax-1

  auto lam_s_data = nda::array<dcomplex, 2>();
  auto lam_d_data = nda::array<dcomplex, 2>();
  auto lam_m_data = nda::array<dcomplex, 2>();
  h5_read(siamdata, "lambda_S", lam_s_data);
  h5_read(siamdata, "lambda_D", lam_d_data);
  h5_read(siamdata, "lambda_M", lam_m_data);

  // Evaluate Green's function on 1D DLR grid and obtain its DLR coefficients
  auto g = nda::vector<dcomplex>(r);
  auto gr = nda::vector<dcomplex>(r); // G reversed: G(-i nu_n)
  for (int k = 0; k < r; ++k) {
    g(k) = g_data(g_nmax + dlr_if_fer(k));
    gr(k) = g_data(g_nmax - dlr_if_fer(k) - 1);
  }
  auto gc = nda::array<dcomplex, 1>(ifops_fer.vals2coefs(beta, g));
  auto grc = nda::array<dcomplex, 1>(ifops_fer.vals2coefs(beta, gr));

  // Evaluate correlation and vertex functions on 2D DLR grid and obtain DLR
  // coefficients
  auto chi_s = nda::vector<dcomplex>(niom);
  auto chi_d = nda::vector<dcomplex>(niom);
  auto chi_m = nda::vector<dcomplex>(niom);
  auto lam_s = nda::vector<dcomplex>(niom);
  auto lam_d = nda::vector<dcomplex>(niom);
  auto lam_m = nda::vector<dcomplex>(niom);
  for (int k = 0; k < niom; ++k) {
    chi_s(k) = chi_s_data(chi_nmax + dlr2d_if(k, 0), chi_nmax + dlr2d_if(k, 1));
    chi_d(k) =
        chi_d_data(chi_nmax + dlr2d_if_ph(k, 0), chi_nmax + dlr2d_if_ph(k, 1));
    chi_m(k) =
        chi_m_data(chi_nmax + dlr2d_if_ph(k, 0), chi_nmax + dlr2d_if_ph(k, 1));
    lam_s(k) =
        lam_s_data(chi_nmax + dlr2d_if(k, 0), chi_nmax + dlr2d_if(k, 1)) - 1;
    lam_d(k) =
        lam_d_data(chi_nmax + dlr2d_if_ph(k, 0), chi_nmax + dlr2d_if_ph(k, 1)) -
        1;
    lam_m(k) =
        lam_m_data(chi_nmax + dlr2d_if_ph(k, 0), chi_nmax + dlr2d_if_ph(k, 1)) -
        1;
  }

  fmt::print("Obtaining DLR coefficients of chi, lambda...\n");
  auto start = std::chrono::high_resolution_clock::now();

  auto valsall = nda::array<dcomplex, 2, F_layout>(niom, 6);
  valsall(_, 0) = chi_s;
  valsall(_, 1) = chi_d;
  valsall(_, 2) = chi_m;
  valsall(_, 3) = lam_s;
  valsall(_, 4) = lam_d;
  valsall(_, 5) = lam_m;

  auto [coefsall, coefsingall] = vals2coefs_if_many_3term(kmat, valsall, r);

  auto chi_s_c = coefsall(0, _, _, _);
  auto chi_d_c = coefsall(1, _, _, _);
  auto chi_m_c = coefsall(2, _, _, _);
  auto lam_s_c = coefsall(3, _, _, _);
  auto lam_d_c = coefsall(4, _, _, _);
  auto lam_m_c = coefsall(5, _, _, _);
  auto chi_s_csing = coefsingall(0, _);
  auto chi_d_csing = coefsingall(1, _);
  auto chi_m_csing = coefsingall(2, _);
  auto lam_s_csing = coefsingall(3, _);
  auto lam_d_csing = coefsingall(4, _);
  auto lam_m_csing = coefsingall(5, _);

  auto end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n", std::chrono::duration<double>(end - start).count());

  // Test expansion of G
  fmt::print("Testing DLR expansion of Green's function...\n");
  auto g_tst = nda::vector<dcomplex>(niomtst);
  auto g_tru = nda::vector<dcomplex>(niomtst);
  for (int m = -niomtst / 2; m < niomtst / 2; ++m) {
    int midx = niomtst / 2 + m;
    g_tst(midx) = ifops_fer.coefs2eval(beta, gc, m);
    g_tru(midx) = g_data(g_nmax + m);
  }
  fmt::print("L2 norm:    {}\n", sqrt(sum(pow(abs(g_tru), 2))) / beta);
  fmt::print("L2 error:   {}\n", sqrt(sum(pow(abs(g_tru - g_tst), 2))) / beta);

  // Test DLR expansion of vertex function
  fmt::print("Testing DLR expansion of vertex function...\n");

  // Evaluate expansion on test grid and measure error
  auto chi_s_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto chi_d_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto chi_m_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto chi_s_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto chi_d_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto chi_m_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_s_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_d_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_m_tst = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_s_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_d_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  auto lam_m_tru = nda::array<dcomplex, 2>(niomtst, niomtst);
  int midx = 0, nidx = 0;
  start = std::chrono::high_resolution_clock::now();
  for (int m = -niomtst / 2; m < niomtst / 2; ++m) {
    for (int n = -niomtst / 2; n < niomtst / 2; ++n) {
      midx = niomtst / 2 + m;
      nidx = niomtst / 2 + n;

      // Evaluate true functions
      chi_s_tru(midx, nidx) = chi_s_data(chi_nmax + m, chi_nmax + n);
      chi_d_tru(midx, nidx) = chi_d_data(chi_nmax + m, chi_nmax + n);
      chi_m_tru(midx, nidx) = chi_m_data(chi_nmax + m, chi_nmax + n);
      lam_s_tru(midx, nidx) = lam_s_data(chi_nmax + m, chi_nmax + n) - 1;
      lam_d_tru(midx, nidx) = lam_d_data(chi_nmax + m, chi_nmax + n) - 1;
      lam_m_tru(midx, nidx) = lam_m_data(chi_nmax + m, chi_nmax + n) - 1;

      // Evaluate DLR expansions
      chi_s_tst(midx, nidx) =
          coefs2eval_if_3term(beta, dlr_rf, chi_s_c, chi_s_csing, m, n, 1);
      chi_d_tst(midx, nidx) =
          coefs2eval_if_3term(beta, dlr_rf, chi_d_c, chi_d_csing, m, n, 2);
      chi_m_tst(midx, nidx) =
          coefs2eval_if_3term(beta, dlr_rf, chi_m_c, chi_m_csing, m, n, 2);
      lam_s_tst(midx, nidx) =
          coefs2eval_if_3term(beta, dlr_rf, lam_s_c, lam_s_csing, m, n, 1);
      lam_d_tst(midx, nidx) =
          coefs2eval_if_3term(beta, dlr_rf, lam_d_c, lam_d_csing, m, n, 2);
      lam_m_tst(midx, nidx) =
          coefs2eval_if_3term(beta, dlr_rf, lam_m_c, lam_m_csing, m, n, 2);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n", std::chrono::duration<double>(end - start).count());

  double chi_s_l2 = sqrt(sum(pow(abs(chi_s_tru), 2))) / beta / beta;
  double chi_d_l2 = sqrt(sum(pow(abs(chi_d_tru), 2))) / beta / beta;
  double chi_m_l2 = sqrt(sum(pow(abs(chi_m_tru), 2))) / beta / beta;
  double lam_s_l2 = sqrt(sum(pow(abs(lam_s_tru), 2))) / beta / beta;
  double lam_d_l2 = sqrt(sum(pow(abs(lam_d_tru), 2))) / beta / beta;
  double lam_m_l2 = sqrt(sum(pow(abs(lam_m_tru), 2))) / beta / beta;

  double chi_s_linf = max_element(abs(chi_s_tru));
  double chi_d_linf = max_element(abs(chi_d_tru));
  double chi_m_linf = max_element(abs(chi_m_tru));
  double lam_s_linf = max_element(abs(lam_s_tru));
  double lam_d_linf = max_element(abs(lam_d_tru));
  double lam_m_linf = max_element(abs(lam_m_tru));

  double chi_s_l2err =
      sqrt(sum(pow(abs(chi_s_tru - chi_s_tst), 2))) / beta / beta;
  double chi_d_l2err =
      sqrt(sum(pow(abs(chi_d_tru - chi_d_tst), 2))) / beta / beta;
  double chi_m_l2err =
      sqrt(sum(pow(abs(chi_m_tru - chi_m_tst), 2))) / beta / beta;
  double lam_s_l2err =
      sqrt(sum(pow(abs(lam_s_tru - lam_s_tst), 2))) / beta / beta;
  double lam_d_l2err =
      sqrt(sum(pow(abs(lam_d_tru - lam_d_tst), 2))) / beta / beta;
  double lam_m_l2err =
      sqrt(sum(pow(abs(lam_m_tru - lam_m_tst), 2))) / beta / beta;

  double chi_s_linferr = max_element(abs(chi_s_tru - chi_s_tst));
  double chi_d_linferr = max_element(abs(chi_d_tru - chi_d_tst));
  double chi_m_linferr = max_element(abs(chi_m_tru - chi_m_tst));
  double lam_s_linferr = max_element(abs(lam_s_tru - lam_s_tst));
  double lam_d_linferr = max_element(abs(lam_d_tru - lam_d_tst));
  double lam_m_linferr = max_element(abs(lam_m_tru - lam_m_tst));

  fmt::print("--- chi_S results ---\n");
  fmt::print("L2 norm:    {}\n", chi_s_l2);
  fmt::print("Linf norm:  {}\n", chi_s_linf);
  fmt::print("L2 error:   {}\n", chi_s_l2err);
  fmt::print("Linf error: {}\n\n", chi_s_linferr);

  fmt::print("--- chi_D results ---\n");
  fmt::print("L2 norm:    {}\n", chi_d_l2);
  fmt::print("Linf norm:  {}\n", chi_d_linf);
  fmt::print("L2 error:   {}\n", chi_d_l2err);
  fmt::print("Linf error: {}\n\n", chi_d_linferr);

  fmt::print("--- chi_M results ---\n");
  fmt::print("L2 norm:    {}\n", chi_m_l2);
  fmt::print("Linf norm:  {}\n", chi_m_linf);
  fmt::print("L2 error:   {}\n", chi_m_l2err);
  fmt::print("Linf error: {}\n\n", chi_m_linferr);

  fmt::print("--- lambda_S results ---\n");
  fmt::print("L2 norm:    {}\n", lam_s_l2);
  fmt::print("Linf norm:  {}\n", lam_s_linf);
  fmt::print("L2 error:   {}\n", lam_s_l2err);
  fmt::print("Linf error: {}\n\n", lam_s_linferr);

  fmt::print("--- lambda_D results ---\n");
  fmt::print("L2 norm:    {}\n", lam_d_l2);
  fmt::print("Linf norm:  {}\n", lam_d_linf);
  fmt::print("L2 error:   {}\n", lam_d_l2err);
  fmt::print("Linf error: {}\n\n", lam_d_linferr);

  fmt::print("--- lambda_M results ---\n");
  fmt::print("L2 norm:    {}\n", lam_m_l2);
  fmt::print("Linf norm:  {}\n", lam_m_linf);
  fmt::print("L2 error:   {}\n", lam_m_l2err);
  fmt::print("Linf error: {}\n\n", lam_m_linferr);

  // Compute polarization from DLR expansions
  auto itops = imtime_ops(lambda, dlr_rf);

  // auto pol_s =
  //     polarization(beta, ifops_fer, ifops_bos, gc, gc, lam_s_c, lam_s_csing);
  // pol_s += polarization_const(beta, itops, ifops_bos, gc, gc);
  auto pol_s = polarization_3term(beta, lambda, eps, itops, ifops_fer, ifops_bos,
                                gc, gc, lam_s_c, lam_s_csing);
  pol_s *= -1.0 / 2;

  // auto pol_d =
  //     polarization(beta, ifops_fer, ifops_bos, grc, gc, lam_d_c,
  //     lam_d_csing);
  // pol_d += polarization_const(beta, itops, ifops_bos, grc, gc);
  auto pol_d =  polarization_3term(beta, lambda, eps, itops, ifops_fer, ifops_bos,
                                grc, gc, lam_d_c, lam_d_csing);

  // auto pol_m =
  //     polarization(beta, ifops_fer, ifops_bos, grc, gc, lam_m_c,
  //     lam_m_csing);
  // pol_m += polarization_const(beta, itops, ifops_bos, grc, gc);
  auto pol_m =  polarization_3term(beta, lambda, eps, itops, ifops_fer, ifops_bos,
                                grc, gc, lam_m_c, lam_m_csing);

  auto pol_s_c = ifops_bos.vals2coefs(beta, pol_s); // DLR expansion
  auto pol_d_c = ifops_bos.vals2coefs(beta, pol_d);
  auto pol_m_c = ifops_bos.vals2coefs(beta, pol_m);

  // Evaluate polarization on dense grid
  auto pol_s_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_d_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_m_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_s_tru = nda::vector<dcomplex>(nbos_tst);
  auto pol_d_tru = nda::vector<dcomplex>(nbos_tst);
  auto pol_m_tru = nda::vector<dcomplex>(nbos_tst);
  std::complex<double> eta_s = 0, eta_d = 0, eta_m = 0;
  for (int n = -nbos_tst / 2; n < nbos_tst / 2; ++n) {
    pol_s_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_s_c, n);
    pol_d_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_d_c, n);
    pol_m_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_m_c, n);

    eta_s = eta_s_data(eta_nmax + n);
    eta_d = eta_d_data(eta_nmax + n);
    eta_m = eta_m_data(eta_nmax + n);

    pol_s_tru(nbos_tst / 2 + n) = (eta_s - 2 * u) / (2 * u * eta_s);
    pol_d_tru(nbos_tst / 2 + n) = (eta_d - u) / (u * eta_d);
    pol_m_tru(nbos_tst / 2 + n) = (eta_m + u) / (-u * eta_m);
  }

  double pol_s_l2 = sqrt(sum(pow(abs(pol_s_tru), 2))) / beta;
  double pol_d_l2 = sqrt(sum(pow(abs(pol_d_tru), 2))) / beta;
  double pol_m_l2 = sqrt(sum(pow(abs(pol_m_tru), 2))) / beta;

  double pol_s_linf = max_element(abs(pol_s_tru));
  double pol_d_linf = max_element(abs(pol_d_tru));
  double pol_m_linf = max_element(abs(pol_m_tru));

  double pol_s_l2err = sqrt(sum(pow(abs(pol_s_tru - pol_s_tst), 2))) / beta;
  double pol_d_l2err = sqrt(sum(pow(abs(pol_d_tru - pol_d_tst), 2))) / beta;
  double pol_m_l2err = sqrt(sum(pow(abs(pol_m_tru - pol_m_tst), 2))) / beta;

  double pol_s_linferr = max_element(abs(pol_s_tru - pol_s_tst));
  double pol_d_linferr = max_element(abs(pol_d_tru - pol_d_tst));
  double pol_m_linferr = max_element(abs(pol_m_tru - pol_m_tst));

  fmt::print("--- pol_s results ---\n");
  fmt::print("L2 norm:    {}\n", pol_s_l2);
  fmt::print("Linf norm:  {}\n", pol_s_linf);
  fmt::print("L2 error:   {}\n", pol_s_l2err);
  fmt::print("Linf error: {}\n\n", pol_s_linferr);

  fmt::print("--- pol_d results ---\n");
  fmt::print("L2 norm:    {}\n", pol_d_l2);
  fmt::print("Linf norm:  {}\n", pol_d_linf);
  fmt::print("L2 error:   {}\n", pol_d_l2err);
  fmt::print("Linf error: {}\n\n", pol_d_linferr);

  fmt::print("--- pol_m results ---\n");
  fmt::print("L2 norm:    {}\n", pol_m_l2);
  fmt::print("Linf norm:  {}\n", pol_m_linf);
  fmt::print("L2 error:   {}\n", pol_m_l2err);
  fmt::print("Linf error: {}\n\n", pol_m_linferr);
  }

int main() {
  double beta = 20; // Inverse temperature
  double u = 5;
  double lambda = 300; // DLR cutoff
  // double eps = 1e-14;   // DLR tolerance
  int niom_dense = 100; // # imag freq sample pts for fine grid (must be even)
  int niomtst = 1024;   // # imag freq test points (must be even)
  int nbos_tst = 1024;  // # pts in test grid for polarization
  bool reduced = true;  // Full or reduced fine grid

  auto filename = "siam_tst1024_new";

  int nexp = 1;
  int nresult = 42;
  auto results = nda::array<double, 2>(nresult, nexp);
  double eps = 0;
  for (int i = 0; i < nexp; ++i) {
    eps = pow(10, -2.0 * (i + 1));
    eps = 1e-12; 
    siam_driver_3term(beta, u, lambda, eps, niom_dense, niomtst,
                            nbos_tst, reduced);
  }
}