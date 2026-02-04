#include "dlr2d.hpp"
#include "utils.hpp"
#include <chrono>
#include <fmt/format.h>
#include <numbers>

namespace cppdlr2d {

  using namespace cppdlr;
  using std::numbers::pi;

  nda::array<int, 2> read_dlr2d_if(std::string path, std::string filename) {
    h5::file file(path + filename, 'r');
    h5::group mygroup(file);
    auto dlr2d_if = h5::read<nda::array<int, 2>>(mygroup, "dlr2d_if");
    return dlr2d_if;
  }

  std::tuple<nda::array<int, 2>, nda::array<int, 2>> read_dlr2d(std::string path, std::string filename) {
    h5::file file(path + filename, 'r');
    h5::group mygroup(file);
    auto dlr2d_if = h5::read<nda::array<int, 2>>(mygroup, "dlr2d_if");
    auto dlr2d_rf = h5::read<nda::array<int, 2>>(mygroup, "dlr2d_rf");
    return {dlr2d_if, dlr2d_rf};
  }

  nda::array<int, 2> build_prod_if(double lambda, nda::vector_const_view<double> dlr_rf) {

    int r = dlr_rf.size(); // # DLR basis functions

    // Get fermionic and bosonic DLR grids
    auto ifops_fer  = imfreq_ops(lambda, dlr_rf, Fermion);
    auto ifops_bos  = imfreq_ops(lambda, dlr_rf, Boson);
    auto dlr_if_fer = ifops_fer.get_ifnodes();
    auto dlr_if_bos = ifops_bos.get_ifnodes();

    auto prod_if = nda::array<int, 2>(3 * r * r, 2);
    for (int m = 0; m < r; ++m) {
      for (int n = 0; n < r; ++n) {
        prod_if(m * r + n, 0) = dlr_if_fer(m); // nu1 = (2*m_j + 1)*i*pi
        prod_if(m * r + n, 1) = dlr_if_fer(n); // nu2 = (2*n_j + 1)*i*pi

        prod_if(r * r + m * r + n, 0) = dlr_if_bos(n) - dlr_if_fer(m) - 1; // nu1 = 2*n_k*i*pi - (2*m_j+1)*i*pi = (2*(n_k-m_j-1)+1)*i*pi
        prod_if(r * r + m * r + n, 1) = dlr_if_fer(m);                     // nu2 = (2*m_j + 1)*i*pi

        prod_if(2 * r * r + m * r + n, 0) = dlr_if_fer(m);                     // nu1 = (2*m_j + 1)*i*pi
        prod_if(2 * r * r + m * r + n, 1) = dlr_if_bos(n) - dlr_if_fer(m) - 1; // nu2 = 2*n_k*i*pi - (2*m_j+1)*i*pi = (2*(n_k-m_j-1)+1)*i*pi
      }
    }

    return prod_if;
  }

  void build_prod_if(double lambda, nda::vector_const_view<double> dlr_rf, const std::string &path, const std::string &filename) {
    auto prod_if = build_prod_if(lambda, dlr_rf);

    // Write prod_if to hdf5 file
    h5::file file(path + filename, 'w');
    h5::group mygroup(file);
    h5::write(mygroup, "prod_if", prod_if);
  }

  std::tuple<nda::array<int, 2>, nda::array<int, 2>> build_dlr2d(double lambda, double eps, bool compressgrid, bool compressbasis) {

    // Get DLR frequencies
    auto dlr_rf = build_dlr_rf(lambda, eps);
    int r       = dlr_rf.size(); // # DLR basis functions

    // Get fine 2D DLR "product" Matsubara frequency grid
    auto prod_if = build_prod_if(lambda, dlr_rf);

    // Get system matrix for fine grid (union of products of shifted DLR grids)
    auto kmat = build_cf2if(1.0, dlr_rf, prod_if);

    auto dlr2d_if = nda::array<int, 2>();

    // TODO: Fill in dlr2d_rf and dlr2d_if with the correct points whether
    // compressgrid/compressbasis are true/false. For example, for dlr2d_rf, if
    // compressbasis is false, just don't subselect. Use the same code as below
    // (taken outside of the if statement), and just take piv = 0, 1, 2, ...,
    // 3*r*r + r.

    int r2d     = 0;
    int nrow    = kmat.shape(0);
    int ncol    = kmat.shape(1);
    auto rf_idx = nda::array<int, 1>();
    if (compressbasis) { // Compress basis using pivoted QR
      auto kmat_copy = fmatrix(kmat);
      auto piv       = nda::zeros<int>(ncol);
      auto tau       = nda::vector<dcomplex>(ncol);
      nda::lapack::geqp3(kmat_copy, piv, tau);
      r2d    = estimate_rank(kmat_copy, eps, 1);
      rf_idx = piv(nda::range(r2d)) - 1; // Take selected frequency pairs
      ncol   = r2d;
    } else {
      rf_idx = nda::arange<int>(ncol); // Take all frequency pairs
    }

    // Extract frequency pairs
    auto dlr2d_rf = nda::array<int, 2>(ncol, 3);
    int idx       = 0;
    int k = 0, l = 0;
    for (int i = 0; i < ncol; ++i) {
      idx = rf_idx(i);
      if (idx < r * r) {
        std::tie(k, l) = ind2sub_c(idx, r);
        dlr2d_rf(i, 0) = 0;
        dlr2d_rf(i, 1) = k;
        dlr2d_rf(i, 2) = l;
      } else if (idx < 2 * r * r) {
        std::tie(k, l) = ind2sub_c(idx - r * r, r);
        dlr2d_rf(i, 0) = 1;
        dlr2d_rf(i, 1) = k;
        dlr2d_rf(i, 2) = l;
      } else if (idx < 3 * r * r) {
        std::tie(k, l) = ind2sub_c(idx - 2 * r * r, r);
        dlr2d_rf(i, 0) = 2;
        dlr2d_rf(i, 1) = k;
        dlr2d_rf(i, 2) = l;
      } else {
        // Singular term: only one real frequency index k
        // Set l = k due to δ_{l,k} factor in the singular term
        dlr2d_rf(i, 0) = 3;
        dlr2d_rf(i, 1) = idx - 3 * r * r;
        dlr2d_rf(i, 2) = idx - 3 * r * r;
      }
    }

    auto if_idx = nda::array<int, 1>();
    if (compressgrid) {

      // Extract selected columns of kmat and transpose
      auto kmat_copy = fmatrix(nrow, ncol);
      for (int j = 0; j < ncol; ++j) { kmat_copy(nda::range::all, j) = kmat(nda::range::all, rf_idx(j)); }
      // auto kmat_copy = make_regular(kmat(nda::range::all, rf_idx)); // TODO:
      // this fails!

      auto kmatt = fmatrix(transpose(kmat_copy));
      auto piv   = nda::zeros<int>(nrow);
      auto tau   = nda::vector<dcomplex>(nrow);
      nda::lapack::geqp3(kmatt, piv, tau);

      if (!compressbasis) { r2d = estimate_rank(kmatt, eps, 1); }

      if_idx = piv(nda::range(r2d)) - 1;
      nrow   = r2d;
    } else {
      if_idx = nda::arange<int>(nrow);
    }

    // Extract imaginary frequency pairs from pivots
    dlr2d_if = nda::array<int, 2>(nrow, 2);
    for (int m = 0; m < nrow; ++m) {
      dlr2d_if(m, 0) = prod_if(if_idx(m), 0);
      dlr2d_if(m, 1) = prod_if(if_idx(m), 1);
    }

    fmt::print("Fine system matrix shape = {} x {}\n", 3 * r * r + r, 3 * r * r + r);
    fmt::print("System matrix rank = {}\n", r2d);
    fmt::print("DLR rank squared = {}\n", r * r);

    return {dlr2d_if, dlr2d_rf};
  }

  void build_dlr2d(double lambda, double eps, const std::string &path, const std::string &filename, bool compressgrid, bool compressbasis) {
    auto [dlr2d_if, dlr2d_rf] = build_dlr2d(lambda, eps, compressgrid, compressbasis);

    // Write to hdf5 file
    h5::file file(path + filename, 'w');
    h5::group mygroup(file);
    h5::write(mygroup, "dlr2d_if", dlr2d_if);
    h5::write(mygroup, "dlr2d_rf", dlr2d_rf);
  }

  // Obtain 2D DLR nodes using reduced fine grid, mixed fermionic/bosonic
  // representation, two terms
  nda::array<int, 2> build_dlr2d_if_3term(double lambda, double eps) {

    int rankmethod = 1;

    // Get DLR frequencies
    auto dlr_rf = build_dlr_rf(lambda, eps);
    int r       = dlr_rf.size(); // # DLR basis functions

    fmt::print("\nDLR cutoff Lambda = {}\n", lambda);
    fmt::print("DLR tolerance epsilon = {}\n", eps);
    fmt::print("# DLR basis functions = {}\n", r);

    // Get fermionic and bosonic DLR grids
    auto ifops_fer  = imfreq_ops(lambda, dlr_rf, Fermion);
    auto ifops_bos  = imfreq_ops(lambda, dlr_rf, Boson);
    auto dlr_if_fer = ifops_fer.get_ifnodes();
    auto dlr_if_bos = ifops_bos.get_ifnodes();

    // Get fine 2D Matsubara frequency sampling grid
    auto nu2didx = nda::array<int, 2>(2 * r * r, 2);
    for (int m = 0; m < r; ++m) {
      for (int n = 0; n < r; ++n) {

        nu2didx(m * r + n, 0) = dlr_if_bos(n) - dlr_if_fer(m) - 1; // nu1 = 2*n_k*i*pi - (2*m_j+1)*i*pi = (2*(n_k-m_j-1)+1)*i*pi
        nu2didx(m * r + n, 1) = dlr_if_fer(m);                     // nu2 = (2*m_j + 1)*i*pi

        nu2didx(r * r + m * r + n, 0) = dlr_if_fer(m);                     // nu1 = (2*m_j + 1)*i*pi
        nu2didx(r * r + m * r + n, 1) = dlr_if_bos(n) - dlr_if_fer(m) - 1; // nu2 = 2*n_k*i*pi - (2*m_j+1)*i*pi = (2*(n_k-m_j-1)+1)*i*pi
      }
    }

    // Get system matrix for dense grid
    auto kmat = fmatrix(2 * r * r, 2 * r * r + r);

    // Regular part
    for (int k = 0; k < r; ++k) {
      for (int l = 0; l < r; ++l) {
        for (int n = 0; n < 2 * r * r; ++n) {

          // kmat(n, r * r + k * r + l) =
          //     k_if(nu2didx(n, 1), dlr_rf(k), Fermion) *
          //     my_k_if_boson(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l));
          // kmat(n, 2 * r * r + k * r + l) =
          //     k_if(nu2didx(n, 0), dlr_rf(k), Fermion) *
          //     my_k_if_boson(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l));
          kmat(n, k * r + l)         = k_if(nu2didx(n, 1), dlr_rf(k), Fermion) * k_if(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l), Boson);
          kmat(n, r * r + k * r + l) = k_if(nu2didx(n, 0), dlr_rf(k), Fermion) * k_if(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l), Boson);

          // kmat(n, k * r + l) =
          //     ker(nu2d(n, 0), dlr_rf(k)) * ker(nu2d(n, 1), dlr_rf(l));
          // kmat(n, r * r + k * r + l) = ker(nu2d(n, 1), dlr_rf(k)) *
          //                              ker(nu2d(n, 0) + nu2d(n, 1), dlr_rf(l));
          // kmat(n, 2 * r * r + k * r + l) =
          //     ker(nu2d(n, 0), dlr_rf(k)) *
          //     ker(nu2d(n, 0) + nu2d(n, 1), dlr_rf(l));
        }
      }
    }

    // Singular part
    for (int k = 0; k < r; ++k) {
      for (int n = 0; n < 2 * r * r; ++n) {
        if (nu2didx(n, 0) == -nu2didx(n, 1) - 1) {
          kmat(n, 2 * r * r + k) = k_if(nu2didx(n, 0), dlr_rf(k), Fermion);
        } else {
          kmat(n, 2 * r * r + k) = 0;
        }
        // if (nu2d(n, 0) == -nu2d(n, 1)) {
        //   kmat(n, 3 * r * r + k) = ker(nu2d(n, 0), dlr_rf(k));
        // } else {
        //   kmat(n, 3 * r * r + k) = 0;
        // }
      }
    }

    fmt::print("Fine system matrix shape = {} x {}\n", kmat.shape(0), kmat.shape(1));

    // Pivoted QR to determine sampling nodes
    auto kmatt = fmatrix(transpose(kmat));
    auto piv   = nda::zeros<int>(2 * r * r);
    auto tau   = nda::vector<dcomplex>(2 * r * r);
    nda::lapack::geqp3(kmatt, piv, tau);

    // Estimate rank
    int niom_skel = 0;
    if (rankmethod == 1) {
      for (int k = 0; k < 2 * r * r; ++k) {
        if (abs(kmatt(k, k)) < eps) {
          niom_skel = k;
          break;
        }
      }
    } else {
      double errsq = 0;
      for (int k = 2 * r * r - 1; k >= 0; --k) {
        errsq += pow(abs(kmatt(k, k)), 2);
        if (sqrt(errsq) > eps) {
          niom_skel = k;
          break;
        }
      }
    }
    // int niom_skel = estimate_rank(kmatt, eps, 2.0, 100);

    // Extract skeleton nodes from pivots
    auto dlr2d_if = nda::array<int, 2>(niom_skel, 2);
    for (int k = 0; k < niom_skel; ++k) {
      dlr2d_if(k, 0) = nu2didx(piv(k) - 1, 0);
      dlr2d_if(k, 1) = nu2didx(piv(k) - 1, 1);
    }

    fmt::print("DLR rank squared = {}\n", r * r);
    fmt::print("System matrix rank = {}\n\n", niom_skel);

    return dlr2d_if;
  }

  void build_dlr2d_if_3term(double lambda, double eps, std::string path, std::string filename) {
    auto dlr2d_if = build_dlr2d_if_3term(lambda, eps);

    // Write dlr2d_if to hdf5 file
    h5::file file(path + filename, 'w');
    h5::group mygroup(file);
    h5::write(mygroup, "dlr2d_if", dlr2d_if);
  }

  fmatrix build_cf2if(double beta, nda::vector_const_view<double> dlr_rf, nda::array_const_view<int, 2> dlr2d_if) {

    int r    = dlr_rf.size();
    int n_if = dlr2d_if.shape(0);

    // Get system matrix for dense grid
    auto cf2if = fmatrix(n_if, 3 * r * r + r);

    // Regular part
    for (int k = 0; k < r; ++k) {
      for (int l = 0; l < r; ++l) {
        for (int n = 0; n < n_if; ++n) {
          cf2if(n, k * r + l) = beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion) * k_if(dlr2d_if(n, 1), dlr_rf(l), Fermion);
          cf2if(n, r * r + k * r + l) =
             beta * beta * k_if(dlr2d_if(n, 1), dlr_rf(k), Fermion) * k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l));
          cf2if(n, 2 * r * r + k * r + l) =
             beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion) * k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l));
        }
      }
    }

    // Singular part
    for (int k = 0; k < r; ++k) {
      for (int n = 0; n < n_if; ++n) {
        if (dlr2d_if(n, 0) == -dlr2d_if(n, 1) - 1) {
          cf2if(n, 3 * r * r + k) = beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion);
        } else {
          cf2if(n, 3 * r * r + k) = 0;
        }
      }
    }

    return cf2if;
  }

  fmatrix build_cf2if(double beta, nda::vector_const_view<double> dlr_rf, nda::array_const_view<int, 2> dlr2d_if,
                      nda::array_const_view<int, 2> dlr2d_rf) {
    int n_if = dlr2d_if.shape(0);
    int n_rf = dlr2d_rf.shape(0);

    // Get system matrix for dense grid
    auto cf2if = fmatrix(n_if, n_rf);

    // Regular part
    int k = 0, l = 0;
    for (int i = 0; i < n_rf; ++i) {
      k = dlr2d_rf(i, 1);
      l = dlr2d_rf(i, 2);
      for (int n = 0; n < n_if; ++n) {
        if (dlr2d_rf(i, 0) == 0) {
          cf2if(n, i) = beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion) * k_if(dlr2d_if(n, 1), dlr_rf(l), Fermion);
        } else if (dlr2d_rf(i, 0) == 1) {
          cf2if(n, i) = beta * beta * k_if(dlr2d_if(n, 1), dlr_rf(k), Fermion) * k_if(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l), Boson);
        } else if (dlr2d_rf(i, 0) == 2) {
          cf2if(n, i) = beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion) * k_if(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l), Boson);
        } else {
          if (dlr2d_if(n, 0) == -dlr2d_if(n, 1) - 1) {
            cf2if(n, i) = beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion);
          } else {
            cf2if(n, i) = 0;
          }
        }
      }
    }

    return cf2if;
  }

  // two terms K matrix
  fmatrix build_cf2if_3term(double beta, nda::vector<double> dlr_rf, nda::array<int, 2> dlr2d_if) {

    int r         = dlr_rf.size();
    int niom_skel = dlr2d_if.shape(0);

    // Get system matrix for dense grid
    auto kmat = fmatrix(niom_skel, 2 * r * r + r);
    // std::complex<double> nu1 = 0, nu2 = 0;

    // Regular part
    for (int k = 0; k < r; ++k) {
      for (int l = 0; l < r; ++l) {
        for (int n = 0; n < niom_skel; ++n) {

          kmat(n, k * r + l) = beta * beta * k_if(dlr2d_if(n, 1), dlr_rf(k), Fermion) * k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l));
          kmat(n, r * r + k * r + l) =
             beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion) * k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l));
        }
      }
    }

    // Singular part
    for (int k = 0; k < r; ++k) {
      for (int n = 0; n < niom_skel; ++n) {
        // nu1 = (2 * dlr2d_if(n, 0) + 1) * pi * 1i;
        if (dlr2d_if(n, 0) == -dlr2d_if(n, 1) - 1) {
          kmat(n, 2 * r * r + k) = beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion);
          // kmat(n, 3 * r * r + k) = beta * ker(nu1, dlr_rf(k));
        } else {
          kmat(n, 2 * r * r + k) = 0;
        }
      }
    }

    return kmat;
  }

  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>> vals2coefs(int r, fmatrix cf2if, nda::vector_const_view<dcomplex> vals,
                                                                          nda::array_const_view<int, 2> dlr2d_rf) {

    int m              = vals.size();
    int n              = dlr2d_rf.shape(0);
    auto tmp           = nda::array<dcomplex, 1>(std::max(m, n));
    tmp(nda::range(m)) = vals;

    auto s   = nda::vector<double>(m); // Singular values (not needed)
    int rank = 0;                      // Rank (not needed)
    nda::lapack::gelss(cf2if, tmp, s, -1.0, rank);

    auto coefreg = nda::zeros<dcomplex>(3, r, r);
    auto coefsng = nda::zeros<dcomplex>(r);

    for (int i = 0; i < n; ++i) {
      if (dlr2d_rf(i, 0) < 3) { // Regular part
        coefreg(dlr2d_rf(i, 0), dlr2d_rf(i, 1), dlr2d_rf(i, 2)) = tmp(i);
      } else { // Singular part
        coefsng(dlr2d_rf(i, 1)) = tmp(i);
      }
    }

    return {coefreg, coefsng};
  }

  std::tuple<nda::array<dcomplex, 4>, nda::array<dcomplex, 2>>
  vals2coefs_many(int r, fmatrix cf2if, nda::array_const_view<dcomplex, 2, nda::F_layout> vals, nda::array_const_view<int, 2> dlr2d_rf) {

    int m                 = vals.shape(0);
    int nrhs              = vals.shape(1);
    int n                 = dlr2d_rf.shape(0);
    auto tmp              = fmatrix(std::max(m, n), nrhs);
    tmp(nda::range(m), _) = vals;

    auto s   = nda::vector<double>(m); // Singular values (not needed)
    int rank = 0;                      // Rank (not needed)
    nda::lapack::gelss(cf2if, tmp, s, -1.0, rank);

    // Output shape (3, r, r, nrhs) for dimension merging optimization
    auto coefreg = nda::zeros<dcomplex>(3, r, r, nrhs);
    // Output shape (r, nrhs)
    auto coefsng = nda::zeros<dcomplex>(r, nrhs);

    for (int j = 0; j < nrhs; ++j) {
      for (int i = 0; i < n; ++i) {
        if (dlr2d_rf(i, 0) < 3) { // Regular part
          coefreg(dlr2d_rf(i, 0), dlr2d_rf(i, 1), dlr2d_rf(i, 2), j) = tmp(i, j);
        } else { // Singular part
          coefsng(dlr2d_rf(i, 1), j) = tmp(i, j);
        }
      }
    }

    return {coefreg, coefsng};
  }

  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>> vals2coefs_if_3term(fmatrix cf2if, nda::vector_const_view<dcomplex> vals, int r) {

    int m              = vals.size();
    int n              = 2 * r * r + r;
    auto tmp           = nda::array<dcomplex, 1>(n);
    tmp(nda::range(m)) = vals;

    auto s   = nda::vector<double>(m); // Singular values (not needed)
    int rank = 0;                      // Rank (not needed)
    nda::lapack::gelss(cf2if, tmp, s, -1.0, rank);

    auto coefreg                = nda::array<dcomplex, 3>(2, r, r);
    auto coefsng                = nda::array<dcomplex, 1>(r);
    reshape(coefreg, 2 * r * r) = tmp(nda::range(2 * r * r));
    coefsng                     = tmp(nda::range(2 * r * r, 2 * r * r + r));

    return {coefreg, coefsng};
  }

  std::tuple<nda::array<dcomplex, 4>, nda::array<dcomplex, 2>>
  vals2coefs_if_many_3term(fmatrix cf2if, nda::array_const_view<dcomplex, 2, nda::F_layout> vals, int r) {

    int m                 = vals.shape(0);
    int nrhs              = vals.shape(1);
    int n                 = 2 * r * r + r;
    auto tmp              = fmatrix(n, nrhs);
    tmp(nda::range(m), _) = vals;

    auto s   = nda::vector<double>(m); // Singular values (not needed)
    int rank = 0;                      // Rank (not needed)
    nda::lapack::gelss(cf2if, tmp, s, -1.0, rank);

    auto coefreg = nda::array<dcomplex, 4>(nrhs, 2, r, r);
    auto coefsng = nda::array<dcomplex, 2>(nrhs, r);

    for (int j = 0; j < nrhs; ++j) {
      reshape(coefreg(j, _, _, _), 2 * r * r) = tmp(nda::range(2 * r * r), j);
      coefsng(j, _)                           = tmp(nda::range(2 * r * r, 2 * r * r + r), j);
    }

    return {coefreg, coefsng};
  }

  // Evaluate 2D DLR expansion
  // Channel = 1 for particle-particle, = 2 for particle-hole
  std::complex<double> coefs2eval_if(double beta, nda::vector<double> dlr_rf, nda::array_const_view<dcomplex, 3> gc_reg,
                                     nda::array_const_view<dcomplex, 1> gc_sng, int m, int n, int channel) {

    int r = dlr_rf.size(); // # DLR basis functions

    // Make sure coefficient array is 3xrxr
    if (gc_reg.shape(0) != 3) throw std::runtime_error("First dim of coefficient array must be 3.");
    if ((gc_reg.shape(1) != r) || (gc_reg.shape(2) != r))
      throw std::runtime_error(
         "Second and third dims of coefficient array must "
         "be # DLR basis functions r.");

    int mm = 0;
    if (channel == 1) { // Particle-particle channel
      mm = m;
    } else if (channel == 2) { // Particle-hole channel
      mm = -m - 1;
    } else {
      throw std::runtime_error("Invalid channel for dlr2d_coefs2eval.");
    }

    auto kfm = nda::vector<dcomplex>(r);
    auto kfn = nda::vector<dcomplex>(r);
    auto kb  = nda::vector<dcomplex>(r);
    // auto kfmn = nda::vector<dcomplex>(r);
    for (int k = 0; k < r; ++k) {
      kfm(k) = k_if(mm, dlr_rf(k), Fermion);
      kfn(k) = k_if(n, dlr_rf(k), Fermion);
      // kb(k) = my_k_if_boson(mm + n + 1, dlr_rf(k));
      kb(k) = k_if_boson(mm + n + 1, dlr_rf(k));
      // kfmn(k) = ker(2*(mm + n + 1)*1i*pi, dlr_rf(k));
    }

    // Evaluate DLR expansion
    auto g = beta * beta
       * (nda::blas::dot(kfm, matvecmul(gc_reg(0, _, _), kfn)) + nda::blas::dot(kfn, matvecmul(gc_reg(1, _, _), kb))
          + nda::blas::dot(kfm, matvecmul(gc_reg(2, _, _), kb)));

    if (mm + n + 1 == 0) { g += beta * beta * nda::blas::dot(gc_sng, kfm); }

    // std::complex<double> g = 0;
    // for (int k = 0; k < r; ++k) {
    //   for (int l = 0; l < r; ++l) {
    //     g += gc(0, k, l) * kfm(k) * kfn(l) + gc(1, k, l) * kfn(k) * kb(l) +
    //          gc(2, k, l) * kfm(k) * kb(l);
    //   }
    // }

    // // for (int k = 0; k < r; ++k) {
    // //   for (int l = 0; l < r; ++l) {
    // //     g += gc(0, k, l) * kfm(k) * kfn(l) + gc(1, k, l) * kfn(k) * kfmn(l)
    // +
    // //          gc(2, k, l) * kfm(k) * kfmn(l);
    // //   }
    // // }
    // g *= beta * beta;
    // if (mm + n + 1 == 0) {
    //   for (int k = 0; k < r; ++k) {
    //     g += beta * gc_sing(k) * kfm(k);
    //   }
    // }

    return g;
  }

  // Evaluate 2D DLR expansion at multiple points
  // Channel = 1 for particle-particle, = 2 for particle-hole
  nda::vector<dcomplex> coefs2eval_if(double beta, nda::vector<double> dlr_rf, nda::array_const_view<dcomplex, 3> gc_reg,
                                      nda::array_const_view<dcomplex, 1> gc_sng, nda::vector_const_view<int> m, nda::vector_const_view<int> n,
                                      int channel) {

    int r    = dlr_rf.size(); // # DLR basis functions
    int npts = m.size();      // # points to evaluate

    // Make sure coefficient array is 3xrxr
    if (gc_reg.shape(0) != 3) throw std::runtime_error("First dim of coefficient array must be 3.");
    if ((gc_reg.shape(1) != r) || (gc_reg.shape(2) != r))
      throw std::runtime_error(
         "Second and third dims of coefficient array must "
         "be # DLR basis functions r.");

    // Make sure m and n have the same size
    if (n.size() != npts) throw std::runtime_error("m and n must have the same size.");

    // Transform m indices for particle-hole channel
    auto mm = nda::vector<int>(npts);
    if (channel == 1) { // Particle-particle channel
      mm = m;
    } else if (channel == 2) { // Particle-hole channel
      mm = -m - 1;
    } else {
      throw std::runtime_error("Invalid channel for coefs2eval_if.");
    }

    // Build kernel matrices
    auto kfm = nda::array<dcomplex, 2>(npts, r);
    auto kfn = nda::array<dcomplex, 2>(npts, r);
    auto kb  = nda::array<dcomplex, 2>(npts, r);
    for (int i = 0; i < npts; ++i) {
      for (int k = 0; k < r; ++k) {
        kfm(i, k) = k_if(mm(i), dlr_rf(k), Fermion);
        kfn(i, k) = k_if(n(i), dlr_rf(k), Fermion);
        kb(i, k)  = k_if_boson(mm(i) + n(i) + 1, dlr_rf(k));
      }
    }

    // Evaluate DLR expansion at all points
    auto tmp1 = hadamard(matmul(kfm, gc_reg(0, _, _)), kfn);
    auto tmp2 = hadamard(matmul(kfn, gc_reg(1, _, _)), kb);
    auto tmp3 = hadamard(matmul(kfm, gc_reg(2, _, _)), kb);

    auto g = nda::vector<dcomplex>(npts);
    for (int i = 0; i < npts; ++i) {
      g(i) = (beta * beta) * sum(tmp1(i, _) + tmp2(i, _) + tmp3(i, _));

      // Add singular contribution if mm + n + 1 == 0
      if (mm(i) + n(i) + 1 == 0) { g(i) += (beta * beta) * nda::blas::dot(gc_sng, kfm(i, _)); }
    }

    return g;
  }

  // Evaluate multiple 2D DLR expansions at multiple points
  // gc_reg has shape (3, r, r, nbatch), gc_sng has shape (r, nbatch)
  // Channel = 1 for particle-particle, = 2 for particle-hole
  nda::array<dcomplex, 2> coefs2eval_if_many(double beta, nda::vector_const_view<double> dlr_rf, nda::array_const_view<dcomplex, 4> gc_reg,
                                             nda::array_const_view<dcomplex, 2> gc_sng, nda::vector_const_view<int> m, nda::vector_const_view<int> n,
                                             int channel) {

    auto r      = dlr_rf.size();   // # DLR basis functions
    auto npts   = m.size();        // # points to evaluate
    auto nbatch = gc_reg.shape(3); // batch is last dimension

    // Validate input shapes
    if (gc_reg.shape(0) != 3) throw std::runtime_error("First dim of coefficient array must be 3.");
    if ((gc_reg.shape(1) != r) || (gc_reg.shape(2) != r))
      throw std::runtime_error("Second and third dims of coefficient array must be # DLR basis functions r.");
    if (gc_sng.shape(0) != r) throw std::runtime_error("First dim of gc_sng must be # DLR basis functions r.");
    if (gc_sng.shape(1) != nbatch) throw std::runtime_error("Second dim of gc_sng must match batch size.");
    if (n.size() != npts) throw std::runtime_error("m and n must have the same size.");

    // Transform m indices for particle-hole channel
    auto mm = nda::vector<int>(npts);
    if (channel == 1) { // Particle-particle channel
      mm = m;
    } else if (channel == 2) { // Particle-hole channel
      mm = -m - 1;
    } else {
      throw std::runtime_error("Invalid channel for coefs2eval_if_many.");
    }

    // Build kernel matrices
    auto kfm = nda::array<dcomplex, 2>(npts, r);
    auto kfn = nda::array<dcomplex, 2>(npts, r);
    auto kb  = nda::array<dcomplex, 2>(npts, r);
    for (int i = 0; i < npts; ++i) {
      for (int k = 0; k < r; ++k) {
        kfm(i, k) = k_if(mm(i), dlr_rf(k), Fermion);
        kfn(i, k) = k_if(n(i), dlr_rf(k), Fermion);
        kb(i, k)  = k_if_boson(mm(i) + n(i) + 1, dlr_rf(k));
      }
    }

    // === DIMENSION MERGING OPTIMIZATION ===
    // Extract term slices - each gc_reg(t, _, _, _) is contiguous (r, r, nbatch)
    auto gc0 = gc_reg(0, _, _, _);
    auto gc1 = gc_reg(1, _, _, _);
    auto gc2 = gc_reg(2, _, _, _);

    // Reshape to (r, r*nbatch) for matmul - this is a VIEW, no copy
    auto gc0_flat = reshape(gc0, r, r * nbatch);
    auto gc1_flat = reshape(gc1, r, r * nbatch);
    auto gc2_flat = reshape(gc2, r, r * nbatch);

    // 3 matmul calls total (not 3*nbatch)
    auto prod0_flat = matmul(kfm, gc0_flat); // (npts, r*nbatch)
    auto prod1_flat = matmul(kfn, gc1_flat);
    auto prod2_flat = matmul(kfm, gc2_flat);

    // Reshape results to (npts, r, nbatch)
    auto prod0 = reshape(prod0_flat, npts, r, nbatch);
    auto prod1 = reshape(prod1_flat, npts, r, nbatch);
    auto prod2 = reshape(prod2_flat, npts, r, nbatch);

    // Hadamard products and reduction
    auto result    = nda::zeros<dcomplex>(nbatch, npts);
    double beta_sq = beta * beta;

    for (int i = 0; i < npts; ++i) {
      for (int l = 0; l < r; ++l) {
        auto kfn_il = kfn(i, l);
        auto kb_il  = kb(i, l);
        for (int j = 0; j < nbatch; ++j) { result(j, i) += beta_sq * (prod0(i, l, j) * kfn_il + (prod1(i, l, j) + prod2(i, l, j)) * kb_il); }
      }
    }

    // Singular contribution: gc_sng (r, nbatch), kfm (npts, r)
    // transpose(gc_sng) @ transpose(kfm) = (nbatch, r) @ (r, npts) -> (nbatch, npts)
    auto result_sng = matmul(transpose(gc_sng), transpose(kfm));
    for (int i = 0; i < npts; ++i) {
      if (mm(i) + n(i) + 1 == 0) {
        for (int j = 0; j < nbatch; ++j) { result(j, i) += beta_sq * result_sng(j, i); }
      }
    }

    return result;
  }

  // Evaluate multiple 2D DLR expansions on a 2D frequency grid
  // gc_reg has shape (3, r, r, nbatch), gc_sng has shape (r, nbatch)
  // Channel = 1 for particle-particle, = 2 for particle-hole
  nda::array<dcomplex, 3> coefs2eval_if_grid(double beta, nda::vector_const_view<double> dlr_rf, nda::array_const_view<dcomplex, 4> gc_reg,
                                             nda::array_const_view<dcomplex, 2> gc_sng, int m_min, int m_max, int n_min, int n_max, int channel) {

    auto nm     = m_max - m_min + 1;
    auto nn     = n_max - n_min + 1;
    auto npts   = nm * nn;
    auto nbatch = gc_reg.shape(3); // batch is last dimension

    // Build flattened m and n vectors from grid ranges
    auto m_vec = nda::vector<int>(npts);
    auto n_vec = nda::vector<int>(npts);
    for (int im = 0; im < nm; ++im) {
      for (int in = 0; in < nn; ++in) {
        m_vec(im * nn + in) = m_min + im;
        n_vec(im * nn + in) = n_min + in;
      }
    }

    // Call coefs2eval_if_many and reshape from (nbatch, nm*nn) to (nbatch, nm, nn)
    auto result_flat = coefs2eval_if_many(beta, dlr_rf, gc_reg, gc_sng, m_vec, n_vec, channel);
    return nda::array<dcomplex, 3>(reshape(result_flat, nbatch, nm, nn));
  }

  // Evaluate 2D DLR expansion with two terms
  // Channel = 1 for particle-particle, = 2 for particle-hole
  std::complex<double> coefs2eval_if_3term(double beta, nda::vector<double> dlr_rf, nda::array_const_view<dcomplex, 3> gc_reg,
                                           nda::array_const_view<dcomplex, 1> gc_sng, int m, int n, int channel) {

    int r = dlr_rf.size(); // # DLR basis functions

    // Make sure coefficient array is 3xrxr
    if (gc_reg.shape(0) != 2) throw std::runtime_error("First dim of coefficient array must be 2.");
    if ((gc_reg.shape(1) != r) || (gc_reg.shape(2) != r))
      throw std::runtime_error(
         "Second and third dims of coefficient array must "
         "be # DLR basis functions r.");

    int mm = 0;
    if (channel == 1) { // Particle-particle channel
      mm = m;
    } else if (channel == 2) { // Particle-hole channel
      mm = -m - 1;
    } else {
      throw std::runtime_error("Invalid channel for dlr2d_coefs2eval.");
    }

    auto kfm = nda::vector<dcomplex>(r);
    auto kfn = nda::vector<dcomplex>(r);
    auto kb  = nda::vector<dcomplex>(r);
    // auto kfmn = nda::vector<dcomplex>(r);
    for (int k = 0; k < r; ++k) {
      kfm(k) = k_if(mm, dlr_rf(k), Fermion);
      kfn(k) = k_if(n, dlr_rf(k), Fermion);
      // kb(k) = my_k_if_boson(mm + n + 1, dlr_rf(k));
      kb(k) = k_if_boson(mm + n + 1, dlr_rf(k));
      // kfmn(k) = ker(2*(mm + n + 1)*1i*pi, dlr_rf(k));
    }

    // Evaluate DLR expansion
    auto g = beta * beta * (nda::blas::dot(kfn, matvecmul(gc_reg(0, _, _), kb)) + nda::blas::dot(kfm, matvecmul(gc_reg(1, _, _), kb)));

    if (mm + n + 1 == 0) { g += beta * beta * nda::blas::dot(gc_sng, kfm); }

    return g;
  }

} // namespace cppdlr2d
