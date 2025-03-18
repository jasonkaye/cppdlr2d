#include "hubatom.hpp"
#include "../src/utils.hpp"
#include <fmt/format.h>

using namespace dlr2d;

nda::vector<double> hubatom_allfuncs(double beta, double u, double lambda,
                                     double eps, int niomtst, int nbos_tst,
                                     bool reduced, bool compressbasis,
                                     int niom_dense) {

  auto path = "../../../dlr2d_if_data/"; // Path for DLR 2D grid data

  // Read 2D DLR grid indices from file
  std::string filename;
  if (!reduced) {
    filename = get_filename(lambda, eps, niom_dense);
  } else {
    filename = get_filename(lambda, eps, compressbasis);
  }

  auto dlr2d_if = nda::array<int, 2>();
  auto dlr2d_rfidx = nda::array<int, 2>();
  if (!compressbasis) {
    dlr2d_if = read_dlr2d_if(path, filename);
  } else {
    std::tie(dlr2d_rfidx, dlr2d_if) = read_dlr2d_rfif(path, filename);
  }

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

  // Build kernel matrix
  auto kmat = nda::matrix<dcomplex>();
  if (!compressbasis) {
    kmat = build_cf2if(beta, dlr_rf, dlr2d_if);
  } else {
    kmat = build_cf2if_square(beta, dlr_rf, dlr2d_rfidx, dlr2d_if);
  }
  int niom = dlr2d_if.shape(0);

  if (!reduced) {
    fmt::print("Fine system matrix shape = {} x {}\n", niom_dense * niom_dense,
               3 * r * r + r);
  } else {
    fmt::print("Fine system matrix shape = {} x {}\n", 3 * r * r,
               3 * r * r + r);
  }

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

  // Evaluate correlation and vertex functions on 2D DLR grid and obtain DLR
  // coefficients
  auto chi_s = nda::vector<dcomplex>(niom);
  auto chi_d = nda::vector<dcomplex>(niom);
  auto chi_m = nda::vector<dcomplex>(niom);
  auto lam_s = nda::vector<dcomplex>(niom);
  auto lam_d = nda::vector<dcomplex>(niom);
  auto lam_m = nda::vector<dcomplex>(niom);
  for (int k = 0; k < niom; ++k) {
    nu1 = (2 * dlr2d_if(k, 0) + 1) * pi * 1i / beta;
    nu2 = (2 * dlr2d_if(k, 1) + 1) * pi * 1i / beta;
    chi_s(k) = chi_s_fun(u, beta, nu1, nu2);
    lam_s(k) = lam_s_fun(u, beta, nu1, nu2);
  }
  for (int k = 0; k < niom; ++k) {
    nu1 = (2 * dlr2d_if_ph(k, 0) + 1) * pi * 1i / beta;
    nu2 = (2 * dlr2d_if_ph(k, 1) + 1) * pi * 1i / beta;
    chi_d(k) = chi_d_fun(u, beta, nu1, nu2);
    chi_m(k) = chi_m_fun(u, beta, nu1, nu2);
    lam_d(k) = lam_d_fun(u, beta, nu1, nu2);
    lam_m(k) = lam_m_fun(u, beta, nu1, nu2);
  }

  fmt::print("Obtaining DLR coefficients of chi, lambda...\n");
  auto chi_s_c = nda::array<dcomplex, 3>();
  auto chi_d_c = nda::array<dcomplex, 3>();
  auto chi_m_c = nda::array<dcomplex, 3>();
  auto lam_s_c = nda::array<dcomplex, 3>();
  auto lam_d_c = nda::array<dcomplex, 3>();
  auto lam_m_c = nda::array<dcomplex, 3>();
  auto chi_s_csing = nda::array<dcomplex, 1>();
  auto chi_d_csing = nda::array<dcomplex, 1>();
  auto chi_m_csing = nda::array<dcomplex, 1>();
  auto lam_s_csing = nda::array<dcomplex, 1>();
  auto lam_d_csing = nda::array<dcomplex, 1>();
  auto lam_m_csing = nda::array<dcomplex, 1>();

  auto start = std::chrono::high_resolution_clock::now();
  if (!compressbasis) {
    auto valsall  = fmatrix(niom, 6);
    valsall(_, 0) = chi_s;
    valsall(_, 1) = chi_d;
    valsall(_, 2) = chi_m;
    valsall(_, 3) = lam_s;
    valsall(_, 4) = lam_d;
    valsall(_, 5) = lam_m;

    auto [coefsall, coefsingall] = vals2coefs_if_many(kmat, valsall, r);

    chi_s_c = coefsall(0, _, _, _);
    chi_d_c = coefsall(1, _, _, _);
    chi_m_c = coefsall(2, _, _, _);
    lam_s_c = coefsall(3, _, _, _);
    lam_d_c = coefsall(4, _, _, _);
    lam_m_c = coefsall(5, _, _, _);
    chi_s_csing = coefsingall(0, _);
    chi_d_csing = coefsingall(1, _);
    chi_m_csing = coefsingall(2, _);
    lam_s_csing = coefsingall(3, _);
    lam_d_csing = coefsingall(4, _);
    lam_m_csing = coefsingall(5, _);

  } else {
    auto chi_s_c_compressed = vals2coefs_if_square(kmat, chi_s);
    auto chi_d_c_compressed = vals2coefs_if_square(kmat, chi_d);
    auto chi_m_c_compressed = vals2coefs_if_square(kmat, chi_m);
    auto lam_s_c_compressed = vals2coefs_if_square(kmat, lam_s);
    auto lam_d_c_compressed = vals2coefs_if_square(kmat, lam_d);
    auto lam_m_c_compressed = vals2coefs_if_square(kmat, lam_m);

    std::tie(chi_s_c, chi_s_csing) =
        uncompress_basis(r, dlr2d_rfidx, chi_s_c_compressed);
    std::tie(chi_d_c, chi_d_csing) =
        uncompress_basis(r, dlr2d_rfidx, chi_d_c_compressed);
    std::tie(chi_m_c, chi_m_csing) =
        uncompress_basis(r, dlr2d_rfidx, chi_m_c_compressed);
    std::tie(lam_s_c, lam_s_csing) =
        uncompress_basis(r, dlr2d_rfidx, lam_s_c_compressed);
    std::tie(lam_d_c, lam_d_csing) =
        uncompress_basis(r, dlr2d_rfidx, lam_d_c_compressed);
    std::tie(lam_m_c, lam_m_csing) =
        uncompress_basis(r, dlr2d_rfidx, lam_m_c_compressed);
  }
  auto end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n", std::chrono::duration<double>(end - start).count());

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
      nu1 = ((2 * m + 1) * pi * 1i) / beta;
      nu2 = ((2 * n + 1) * pi * 1i) / beta;
      midx = niomtst / 2 + m;
      nidx = niomtst / 2 + n;

      // Evaluate true functions
      chi_s_tru(midx, nidx) = chi_s_fun(u, beta, nu1, nu2);
      chi_d_tru(midx, nidx) = chi_d_fun(u, beta, nu1, nu2);
      chi_m_tru(midx, nidx) = chi_m_fun(u, beta, nu1, nu2);
      lam_s_tru(midx, nidx) = lam_s_fun(u, beta, nu1, nu2);
      lam_d_tru(midx, nidx) = lam_d_fun(u, beta, nu1, nu2);
      lam_m_tru(midx, nidx) = lam_m_fun(u, beta, nu1, nu2);

      // Evaluate DLR expansions
      chi_s_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, chi_s_c, chi_s_csing, m, n, 1);
      chi_d_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, chi_d_c, chi_d_csing, m, n, 2);
      chi_m_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, chi_m_c, chi_m_csing, m, n, 2);
      lam_s_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, lam_s_c, lam_s_csing, m, n, 1);
      lam_d_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, lam_d_c, lam_d_csing, m, n, 2);
      lam_m_tst(midx, nidx) =
          coefs2eval_if(beta, dlr_rf, lam_m_c, lam_m_csing, m, n, 2);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n\n",
             std::chrono::duration<double>(end - start).count());

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
  //     polarization_res(beta, ifops_fer, ifops_bos, gc, gc, lam_s_c,
  //     lam_s_csing);
  // pol_s += polarization_const(beta, itops, ifops_bos, gc, gc);
  auto pol_s = polarization(beta, lambda, eps, itops, ifops_fer, ifops_bos, gc,
                            gc, lam_s_c, lam_s_csing);
  pol_s *= -1.0 / 2;

  // auto pol_d =
  //     polarization_res(beta, ifops_fer, ifops_bos, grc, gc, lam_d_c,
  //     lam_d_csing);
  // pol_d += polarization_const(beta, itops, ifops_bos, grc, gc);
  auto pol_d = polarization(beta, lambda, eps, itops, ifops_fer, ifops_bos, grc,
                            gc, lam_d_c, lam_d_csing);

  // auto pol_m =
  //     polarization_res(beta, ifops_fer, ifops_bos, grc, gc, lam_m_c,
  //     lam_m_csing);
  // pol_m += polarization_const(beta, itops, ifops_bos, grc, gc);
  auto pol_m = polarization(beta, lambda, eps, itops, ifops_fer, ifops_bos, grc,
                            gc, lam_m_c, lam_m_csing);

  auto pol_s_c = ifops_bos.vals2coefs(beta, pol_s); // DLR expansion
  auto pol_d_c = ifops_bos.vals2coefs(beta, pol_d);
  auto pol_m_c = ifops_bos.vals2coefs(beta, pol_m);

  // Compute true polarization
  std::complex<double> pol0_s_tru =
      beta * -k_it(0.0, -u / 2, beta) /
      (2 * beta * u * -k_it(0.0, -u / 2, beta) - 4);
  std::complex<double> pol0_d_tru = beta * -k_it(0.0, -u / 2, beta) /
                                    (beta * u * -k_it(0.0, -u / 2, beta) - 2);
  std::complex<double> pol0_m_tru = beta * -k_it(0.0, u / 2, beta) /
                                    (-beta * u * -k_it(0.0, u / 2, beta) - 2);

  // Evaluate polarization on dense grid
  auto pol_s_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_d_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_m_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_s_tru = nda::vector<dcomplex>(nbos_tst);
  auto pol_d_tru = nda::vector<dcomplex>(nbos_tst);
  auto pol_m_tru = nda::vector<dcomplex>(nbos_tst);
  for (int n = -nbos_tst / 2; n < nbos_tst / 2; ++n) {
    pol_s_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_s_c, n);
    pol_d_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_d_c, n);
    pol_m_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_m_c, n);

    if (n == 0) {
      pol_s_tru(n + nbos_tst / 2) = pol0_s_tru;
      pol_d_tru(n + nbos_tst / 2) = pol0_d_tru;
      pol_m_tru(n + nbos_tst / 2) = pol0_m_tru;
    } else {
      pol_s_tru(n + nbos_tst / 2) = 0;
      pol_d_tru(n + nbos_tst / 2) = 0;
      pol_m_tru(n + nbos_tst / 2) = 0;
    }
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

  auto results = nda::vector<double>(42);
  results(0) = beta;
  results(1) = u;
  results(2) = lambda;
  results(3) = eps;
  results(4) = r;
  results(5) = niom;
  results(6) = chi_s_l2;
  results(7) = chi_s_linf;
  results(8) = chi_s_l2err;
  results(9) = chi_s_linferr;
  results(10) = chi_d_l2;
  results(11) = chi_d_linf;
  results(12) = chi_d_l2err;
  results(13) = chi_d_linferr;
  results(14) = chi_m_l2;
  results(15) = chi_m_linf;
  results(16) = chi_m_l2err;
  results(17) = chi_m_linferr;
  results(18) = lam_s_l2;
  results(19) = lam_s_linf;
  results(20) = lam_s_l2err;
  results(21) = lam_s_linferr;
  results(22) = lam_d_l2;
  results(23) = lam_d_linf;
  results(24) = lam_d_l2err;
  results(25) = lam_d_linferr;
  results(26) = lam_m_l2;
  results(27) = lam_m_linf;
  results(28) = lam_m_l2err;
  results(29) = lam_m_linferr;
  results(30) = pol_s_l2;
  results(31) = pol_s_linf;
  results(32) = pol_s_l2err;
  results(33) = pol_s_linferr;
  results(34) = pol_d_l2;
  results(35) = pol_d_linf;
  results(36) = pol_d_l2err;
  results(37) = pol_d_linferr;
  results(38) = pol_m_l2;
  results(39) = pol_m_linf;
  results(40) = pol_m_l2err;
  results(41) = pol_m_linferr;

  // // Output polarization to file
  // std::ofstream f1("pol_tst.dat");
  // std::ofstream f2("pol_tru.dat");
  // f1.precision(16);
  // f2.precision(16);
  // for (int i = 0; i < nbos_tst; ++i) {
  //   f1 << pol_d_tst(i).real() << " " << pol_d_tst(i).imag() << "\n";
  //   f2 << pol_d_tru(i).real() << " " << pol_d_tru(i).imag() << "\n";
  // }
  // f1.close();
  // f2.close();

  // PRINT(make_regular(abs(pol_m_tru - pol_m_tst)));
  return results;
}

nda::vector<double> hubatom_allfuncs_3term(double beta, double u, double lambda,
                                           double eps, int niomtst,
                                           int nbos_tst) {

  auto path = "../../../dlr2d_if_data/"; // Path for DLR 2D grid data

  // Read 2D DLR grid indices from file
  std::string filename;

  filename = get_filename_3term(lambda, eps);

  auto dlr2d_if = nda::array<int, 2>();
  auto dlr2d_rfidx = nda::array<int, 2>();
  dlr2d_if = read_dlr2d_if(path, filename);

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

  // Build kernel matrix
  auto kmat = nda::matrix<dcomplex>();
  kmat = build_cf2if_3term(beta, dlr_rf, dlr2d_if);
  int niom = dlr2d_if.shape(0);

  fmt::print("Fine system matrix shape = {} x {}\n", 2 * r * r, 2 * r * r + r);

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

  // Evaluate correlation and vertex functions on 2D DLR grid and obtain DLR
  // coefficients
  auto chi_s = nda::vector<dcomplex>(niom);
  auto chi_d = nda::vector<dcomplex>(niom);
  auto chi_m = nda::vector<dcomplex>(niom);
  auto lam_s = nda::vector<dcomplex>(niom);
  auto lam_d = nda::vector<dcomplex>(niom);
  auto lam_m = nda::vector<dcomplex>(niom);
  for (int k = 0; k < niom; ++k) {
    nu1 = (2 * dlr2d_if(k, 0) + 1) * pi * 1i / beta;
    nu2 = (2 * dlr2d_if(k, 1) + 1) * pi * 1i / beta;
    chi_s(k) = chi_s_fun(u, beta, nu1, nu2);
    lam_s(k) = lam_s_fun(u, beta, nu1, nu2);
  }
  for (int k = 0; k < niom; ++k) {
    nu1 = (2 * dlr2d_if_ph(k, 0) + 1) * pi * 1i / beta;
    nu2 = (2 * dlr2d_if_ph(k, 1) + 1) * pi * 1i / beta;
    chi_d(k) = chi_d_fun(u, beta, nu1, nu2);
    chi_m(k) = chi_m_fun(u, beta, nu1, nu2);
    lam_d(k) = lam_d_fun(u, beta, nu1, nu2);
    lam_m(k) = lam_m_fun(u, beta, nu1, nu2);
  }

  fmt::print("Obtaining DLR coefficients of chi, lambda...\n");
  auto chi_s_c = nda::array<dcomplex, 3>();
  auto chi_d_c = nda::array<dcomplex, 3>();
  auto chi_m_c = nda::array<dcomplex, 3>();
  auto lam_s_c = nda::array<dcomplex, 3>();
  auto lam_d_c = nda::array<dcomplex, 3>();
  auto lam_m_c = nda::array<dcomplex, 3>();
  auto chi_s_csing = nda::array<dcomplex, 1>();
  auto chi_d_csing = nda::array<dcomplex, 1>();
  auto chi_m_csing = nda::array<dcomplex, 1>();
  auto lam_s_csing = nda::array<dcomplex, 1>();
  auto lam_d_csing = nda::array<dcomplex, 1>();
  auto lam_m_csing = nda::array<dcomplex, 1>();

  auto start = std::chrono::high_resolution_clock::now();
  auto valsall  = fmatrix(niom, 6);
  valsall(_, 0) = chi_s;
  valsall(_, 1) = chi_d;
  valsall(_, 2) = chi_m;
  valsall(_, 3) = lam_s;
  valsall(_, 4) = lam_d;
  valsall(_, 5) = lam_m;

  auto [coefsall, coefsingall] = vals2coefs_if_many_3term(kmat, valsall, r);

  chi_s_c = coefsall(0, _, _, _);
  chi_d_c = coefsall(1, _, _, _);
  chi_m_c = coefsall(2, _, _, _);
  lam_s_c = coefsall(3, _, _, _);
  lam_d_c = coefsall(4, _, _, _);
  lam_m_c = coefsall(5, _, _, _);
  chi_s_csing = coefsingall(0, _);
  chi_d_csing = coefsingall(1, _);
  chi_m_csing = coefsingall(2, _);
  lam_s_csing = coefsingall(3, _);
  lam_d_csing = coefsingall(4, _);
  lam_m_csing = coefsingall(5, _);
  auto end = std::chrono::high_resolution_clock::now();
  fmt::print("Time: {}\n", std::chrono::duration<double>(end - start).count());

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
      nu1 = ((2 * m + 1) * pi * 1i) / beta;
      nu2 = ((2 * n + 1) * pi * 1i) / beta;
      midx = niomtst / 2 + m;
      nidx = niomtst / 2 + n;

      // Evaluate true functions
      chi_s_tru(midx, nidx) = chi_s_fun(u, beta, nu1, nu2);
      chi_d_tru(midx, nidx) = chi_d_fun(u, beta, nu1, nu2);
      chi_m_tru(midx, nidx) = chi_m_fun(u, beta, nu1, nu2);
      lam_s_tru(midx, nidx) = lam_s_fun(u, beta, nu1, nu2);
      lam_d_tru(midx, nidx) = lam_d_fun(u, beta, nu1, nu2);
      lam_m_tru(midx, nidx) = lam_m_fun(u, beta, nu1, nu2);

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
  fmt::print("Time: {}\n\n",
             std::chrono::duration<double>(end - start).count());

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
  //     polarization_res(beta, ifops_fer, ifops_bos, gc, gc, lam_s_c,
  //     lam_s_csing);
  // pol_s += polarization_const(beta, itops, ifops_bos, gc, gc);
  auto pol_s =
      dlr2d::polarization_3term(beta, lambda, eps, itops, ifops_fer, ifops_bos,
                                gc, gc, lam_s_c, lam_s_csing);
  pol_s *= -1.0 / 2;

  // auto pol_d =
  //     polarization_res(beta, ifops_fer, ifops_bos, grc, gc, lam_d_c,
  //     lam_d_csing);
  // pol_d += polarization_const(beta, itops, ifops_bos, grc, gc);
  auto pol_d =
      dlr2d::polarization_3term(beta, lambda, eps, itops, ifops_fer, ifops_bos,
                                grc, gc, lam_d_c, lam_d_csing);

  // auto pol_m =
  //     polarization_res(beta, ifops_fer, ifops_bos, grc, gc, lam_m_c,
  //     lam_m_csing);
  // pol_m += polarization_const(beta, itops, ifops_bos, grc, gc);
  auto pol_m =
      dlr2d::polarization_3term(beta, lambda, eps, itops, ifops_fer, ifops_bos,
                                grc, gc, lam_m_c, lam_m_csing);

  auto pol_s_c = ifops_bos.vals2coefs(beta, pol_s); // DLR expansion
  auto pol_d_c = ifops_bos.vals2coefs(beta, pol_d);
  auto pol_m_c = ifops_bos.vals2coefs(beta, pol_m);

  // Compute true polarization
  std::complex<double> pol0_s_tru =
      beta * -k_it(0.0, -u / 2, beta) /
      (2 * beta * u * -k_it(0.0, -u / 2, beta) - 4);
  std::complex<double> pol0_d_tru = beta * -k_it(0.0, -u / 2, beta) /
                                    (beta * u * -k_it(0.0, -u / 2, beta) - 2);
  std::complex<double> pol0_m_tru = beta * -k_it(0.0, u / 2, beta) /
                                    (-beta * u * -k_it(0.0, u / 2, beta) - 2);

  // Evaluate polarization on dense grid
  auto pol_s_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_d_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_m_tst = nda::vector<dcomplex>(nbos_tst);
  auto pol_s_tru = nda::vector<dcomplex>(nbos_tst);
  auto pol_d_tru = nda::vector<dcomplex>(nbos_tst);
  auto pol_m_tru = nda::vector<dcomplex>(nbos_tst);
  for (int n = -nbos_tst / 2; n < nbos_tst / 2; ++n) {
    pol_s_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_s_c, n);
    pol_d_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_d_c, n);
    pol_m_tst(n + nbos_tst / 2) = ifops_bos.coefs2eval(beta, pol_m_c, n);

    if (n == 0) {
      pol_s_tru(n + nbos_tst / 2) = pol0_s_tru;
      pol_d_tru(n + nbos_tst / 2) = pol0_d_tru;
      pol_m_tru(n + nbos_tst / 2) = pol0_m_tru;
    } else {
      pol_s_tru(n + nbos_tst / 2) = 0;
      pol_d_tru(n + nbos_tst / 2) = 0;
      pol_m_tru(n + nbos_tst / 2) = 0;
    }
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

  auto results = nda::vector<double>(42);
  results(0) = beta;
  results(1) = u;
  results(2) = lambda;
  results(3) = eps;
  results(4) = r;
  results(5) = niom;
  results(6) = chi_s_l2;
  results(7) = chi_s_linf;
  results(8) = chi_s_l2err;
  results(9) = chi_s_linferr;
  results(10) = chi_d_l2;
  results(11) = chi_d_linf;
  results(12) = chi_d_l2err;
  results(13) = chi_d_linferr;
  results(14) = chi_m_l2;
  results(15) = chi_m_linf;
  results(16) = chi_m_l2err;
  results(17) = chi_m_linferr;
  results(18) = lam_s_l2;
  results(19) = lam_s_linf;
  results(20) = lam_s_l2err;
  results(21) = lam_s_linferr;
  results(22) = lam_d_l2;
  results(23) = lam_d_linf;
  results(24) = lam_d_l2err;
  results(25) = lam_d_linferr;
  results(26) = lam_m_l2;
  results(27) = lam_m_linf;
  results(28) = lam_m_l2err;
  results(29) = lam_m_linferr;
  results(30) = pol_s_l2;
  results(31) = pol_s_linf;
  results(32) = pol_s_l2err;
  results(33) = pol_s_linferr;
  results(34) = pol_d_l2;
  results(35) = pol_d_linf;
  results(36) = pol_d_l2err;
  results(37) = pol_d_linferr;
  results(38) = pol_m_l2;
  results(39) = pol_m_linf;
  results(40) = pol_m_l2err;
  results(41) = pol_m_linferr;

  return results;
}

// NOTE: for now, unfortunately, nu means i*nu; change this in the future

std::complex<double> g_fun(double u, std::complex<double> nu) {
  return 1.0 / (nu - u * u / (4 * nu));
}

std::complex<double> chi_s_fun(double u, double beta, std::complex<double> nu1,
                               std::complex<double> nu2) {
  std::complex<double> val = 0;

  // Regular part
  val = (2 - u * u / (2 * nu1 * nu2));

  // Singular part
  if (abs(nu1 + nu2) == 0) {
    val -=
        beta * u * -k_it(0.0, -u / 2, beta) * (1.0 - u * u / (4 * nu1 * nu1));
  }

  // Multiply by Pi(nu1,nu2)
  val *= g_fun(u, nu1) * g_fun(u, nu2);

  return val;
}

std::complex<double> chi_ph_fun(double u, double beta, std::complex<double> nu1,
                                std::complex<double> nu2, int channel) {
  double uu = channel * u;
  std::complex<double> val = 0;

  // Regular part
  val = (2 + u * u / (2 * nu1 * nu2));

  // Singular part
  if (abs(nu1 - nu2) == 0) {
    val -=
        beta * uu * -k_it(0.0, -uu / 2, beta) * (1.0 - u * u / (4 * nu1 * nu1));
  }

  // Multiply by -1/2 * Pi(nu1,nu2)
  val *= -g_fun(u, nu1) * g_fun(u, nu2) / 2;

  // If density channel, add to singular part
  if (channel == 1 && abs(nu1 - nu2) == 0) {
    val += beta * g_fun(u, nu2);
  }

  return val;
}

std::complex<double> chi_d_fun(double u, double beta, std::complex<double> nu1,
                               std::complex<double> nu2) {
  return chi_ph_fun(u, beta, nu1, nu2, 1);
}

std::complex<double> chi_m_fun(double u, double beta, std::complex<double> nu1,
                               std::complex<double> nu2) {
  return chi_ph_fun(u, beta, nu1, nu2, -1);
}

std::complex<double> lam_s_fun(double u, double beta, std::complex<double> nu1,
                               std::complex<double> nu2) {
  std::complex<double> val = 0;

  // Regular part of chi/Pi
  val = 2 - u * u / (2 * nu1 * nu2);

  // Singular part of chi/Pi
  if (abs(nu1 + nu2) == 0) {
    val -=
        beta * u * -k_it(0.0, -u / 2, beta) * (1.0 + u * u / (4 * nu1 * nu2));
  }

  // Divide by eta
  if (abs(nu1 + nu2) != 0) {
    val /= 2.0;
  } else {
    val /= 2.0 - beta * u * -k_it(0.0, -u / 2, beta);
  }

  return val - 1.0;
}

std::complex<double> lam_ph_fun(double u, double beta, std::complex<double> nu1,
                                std::complex<double> nu2, int channel) {
  double uu = channel * u;
  std::complex<double> val = 0;

  // Regular part of chi/Pi
  val = 2 + u * u / (2 * nu1 * nu2);

  // Singular part of chi/Pi
  if (abs(nu1 - nu2) == 0) {
    val -=
        beta * uu * -k_it(0.0, -uu / 2, beta) * (1.0 - u * u / (4 * nu1 * nu2));
  }

  // Divide by eta
  if (abs(nu1 - nu2) != 0) {
    val /= 2.0;
  } else {
    val /= 2.0 - beta * uu * -k_it(0.0, -uu / 2, beta);
  }

  return val - 1.0;
}

std::complex<double> lam_d_fun(double u, double beta, std::complex<double> nu1,
                               std::complex<double> nu2) {
  return lam_ph_fun(u, beta, nu1, nu2, 1);
}

std::complex<double> lam_m_fun(double u, double beta, std::complex<double> nu1,
                               std::complex<double> nu2) {
  return lam_ph_fun(u, beta, nu1, nu2, -1);
}
