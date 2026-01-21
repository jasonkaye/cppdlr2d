#include "utils.hpp"

#include <iomanip>

namespace dlr2d {

  nda::array<int, 2> get_dlr2d_if_ph(nda::array_const_view<int, 2> dlr2d_if) {
    auto dlr2d_if_ph  = nda::array<int, 2>(dlr2d_if.shape());
    dlr2d_if_ph(_, 0) = -dlr2d_if(_, 0) - 1;
    dlr2d_if_ph(_, 1) = dlr2d_if(_, 1);
    return dlr2d_if_ph;
  }

  std::string get_filename(double lambda, double eps, int niom_dense) {

    std::ostringstream filenameStream;
    filenameStream << "dlr2d_if_fullgrid_" << lambda << "_" << std::scientific << std::setprecision(2) << eps << "_" << niom_dense << ".h5";
    return filenameStream.str();
  }

  std::string get_filename(double lambda, double eps, bool compressgrid, bool compressbasis) {

    std::ostringstream filenameStream;
    auto str = "";
    if (compressgrid && compressbasis) {
      str = "compressed_if_rf";
    } else if (compressgrid && !compressbasis) {
      str = "compressed_if";
    } else if (!compressgrid && compressbasis) {
      str = "compressed_rf";
    } else {
      str = "overcomplete";
    }

    filenameStream << "dlr2d_" << str << "_" << lambda << "_" << std::scientific << std::setprecision(2) << eps << ".h5";

    return filenameStream.str();
  }

  std::string get_filename_3term(double lambda, double eps) {

    std::ostringstream filenameStream;
    filenameStream << "dlr2d_if_3term_" << lambda << "_" << std::scientific << std::setprecision(2) << eps << ".h5";

    return filenameStream.str();
  }

  int estimate_rank(fmatrix_const_view a, double eps, int method, double alpha, int nvec) {

    int n    = a.shape(1);
    int rank = 0;

    if (method == 1) {
      // Method 1: estimate based on diagonal entries of R
      for (int k = 0; k < n; ++k) {
        if (abs(a(k, k)) < eps) {
          rank = k;
          break;
        }
      }
    } else if (method == 2) {
      // Method 2: estimate based on sum of squares of lower right entries of R
      double errsq = 0;
      for (int k = n - 1; k >= 0; --k) {
        errsq += pow(abs(a(k, k)), 2);
        if (sqrt(errsq) > eps) {
          rank = k;
          break;
        }
      }
    } else if (method == 3) {

      if (!alpha || !nvec) { throw std::invalid_argument("alpha and nvec must be provided for method 3"); }

      // Set up random number generator
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<double> d(0.0, 1.0);

      // Generate random Gaussian vectors
      auto x = nda::matrix<dcomplex>(n, nvec);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nvec; ++j) { x(i, j) = d(gen) + 1i * d(gen); }
      }

      // Extract upper triangular matrix R
      auto r = fmatrix(n, n);
      for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) { r(i, j) = a(i, j); }
      }

      // Multiply R by random vectors
      auto y = r * x;

      // Compute cumulative l2 norms of columns of x, starting from the bottom
      auto xnorm = nda::zeros<double>(x.shape());
      for (int j = 0; j < nvec; ++j) {
        // Compute cumulative sum of squares of elements in column j, starting
        // from bottom
        xnorm(n - 1, j) = pow(abs(x(n - 1, j)), 2);
        for (int i = n - 2; i >= 0; --i) { xnorm(i, j) = pow(abs(x(i, j)), 2) + xnorm(i + 1, j); }
      }

      // Compute cumulative l2 norms of columns of y, starting from the bottom
      auto ynorm = nda::zeros<double>(y.shape());
      for (int j = 0; j < nvec; ++j) {
        // Compute cumulative sum of squares of elements in column j, starting
        // from bottom
        ynorm(n - 1, j) = pow(abs(y(n - 1, j)), 2);
        for (int i = n - 2; i >= 0; --i) { ynorm(i, j) = pow(abs(y(i, j)), 2) + ynorm(i + 1, j); }
      }

      ynorm /= xnorm;

      // Take maximum of cumulative l2 norms over random vectors
      for (int i = 0; i < n; ++i) { ynorm(i, 0) = max_element(ynorm(i, _)); }

      // Estimate rank
      int rank     = 0;
      double epssc = eps / (alpha * sqrt(2 / pi));
      for (int i = 0; i < n - 1; ++i) {
        if (ynorm(i + 1, 0) < epssc * epssc) {
          rank = i + 1;
          break;
        }
      }
    } else {
      throw std::runtime_error("Must have method = 1, 2, or 3.");
    }
    return rank;
  }

  std::complex<double> ker(std::complex<double> nu, double om) { return 1.0 / (nu - om); }

  std::complex<double> my_k_if_boson(int n, double om) { return 1.0 / (2 * n * pi * 1i - om); }

  // Get bosonic DLR Matsubara frequency grid with modified kernel
  nda::vector<int> get_dlr_if_boson(double lambda, nda::vector_const_view<double> dlr_rf) {
    int nmax          = cppdlr::fineparams(lambda).nmax;
    int r             = dlr_rf.size();
    auto dlr_if_boson = nda::vector<int>(r);

    auto kmat = nda::matrix<dcomplex>(2 * nmax + 1, r);

    for (int n = -nmax; n <= nmax; ++n) {
      for (int j = 0; j < r; ++j) { kmat(nmax + n, j) = my_k_if_boson(n, dlr_rf(j)); }
    }

    auto [q, norms, piv] = cppdlr::pivrgs(kmat, 1e-100);
    std::sort(piv.begin(), piv.end()); // Sort pivots in ascending order
    for (int i = 0; i < r; ++i) { dlr_if_boson(i) = piv(i) - nmax; }

    return dlr_if_boson;
  }

  std::tuple<int, int> ind2sub(int idx, int n) {
    if (idx >= n * n) throw std::runtime_error("Index out of bounds.");
    int i = idx % n;
    int j = (idx - i) / n;
    return {i, j};
  }

  // Convert linear index to C-order subscripts
  std::tuple<int, int> ind2sub_c(int idx, int n) {
    if (idx >= n * n) throw std::runtime_error("Index out of bounds.");
    int j = idx % n;
    int i = (idx - j) / n;
    return {i, j};
  }

} // namespace dlr2d
