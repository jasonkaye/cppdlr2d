#include "dlr2d.hpp"
#include "nda/declarations.hpp"
#include "nda/lapack/geqp3.hpp"
#include <fmt/format.h>
#include <fstream>
#include <h5/h5.hpp>
#include <nda/h5.hpp>
#include <numbers>
#include <random>

using namespace cppdlr;
using namespace nda;
using namespace std::numbers;

std::complex<double> ker(std::complex<double> nu, double om) {
  return 1.0 / (nu - om);
}

std::complex<double> my_k_if_boson(int n, double om) {
  return 1.0 / (2 * n * pi * 1i - om);
}

// Convert linear index to Fortran-order subscripts
std::tuple<int, int> ind2sub(int idx, int n) {
  if (idx >= n * n)
    throw std::runtime_error("Index out of bounds.");
  int i = idx % n;
  int j = (idx - i) / n;
  return {i, j};
}

// Convert linear index to C-order subscripts
std::tuple<int, int> ind2sub_c(int idx, int n) {
  if (idx >= n * n)
    throw std::runtime_error("Index out of bounds.");
  int j = idx % n;
  int i = (idx - j) / n;
  return {i, j};
}

// Obtain 2D DLR nodes

void get_dlr2d_if(nda::vector<double> dlr_rf, int niom_dense, double eps,
                  std::string path, std::string filename) {

  int r = dlr_rf.size();

  // Get dense Matsubara frequency sampling grid
  auto nu_dense = nda::vector<dcomplex>(niom_dense);
  for (int n = -niom_dense / 2; n < niom_dense / 2; ++n) {
    nu_dense(n + niom_dense / 2) = (2 * n + 1) * pi * 1i;
  }

  // Get system matrix for dense grid
  auto kmat =
      nda::matrix<dcomplex, F_layout>(niom_dense * niom_dense, 3 * r * r + r);
  std::complex<double> nu1 = 0, nu2 = 0;

  // Regular part
  for (int k = 0; k < r; ++k) {
    for (int l = 0; l < r; ++l) {
      for (int n = 0; n < niom_dense; ++n) {
        for (int m = 0; m < niom_dense; ++m) {
          kmat(n * niom_dense + m, k * r + l) =
              ker(nu_dense(m), dlr_rf(k)) * ker(nu_dense(n), dlr_rf(l));
          kmat(n * niom_dense + m, r * r + k * r + l) =
              ker(nu_dense(n), dlr_rf(k)) *
              ker(nu_dense(m) + nu_dense(n), dlr_rf(l));
          kmat(n * niom_dense + m, 2 * r * r + k * r + l) =
              ker(nu_dense(m), dlr_rf(k)) *
              ker(nu_dense(m) + nu_dense(n), dlr_rf(l));
        }
      }
    }
  }

  // Singular part
  for (int k = 0; k < r; ++k) {
    for (int n = 0; n < niom_dense; ++n) {
      for (int m = 0; m < niom_dense; ++m) {
        if (m == -n - 1) {
          kmat(n * niom_dense + m, 3 * r * r + k) = ker(nu_dense(m), dlr_rf(k));
        } else {
          kmat(n * niom_dense + m, 3 * r * r + k) = 0;
        }
      }
    }
  }

  // // Pivoted QR to determine sampling nodes
  // fmt::print("Pivoted QR to determine skeleton nodes...\n");
  // auto start = std::chrono::high_resolution_clock::now();
  // auto [q, nrm, piv] = pivrgs(kmat, eps);
  // int niom_skel = piv.size();
  // auto end = std::chrono::high_resolution_clock::now();
  // fmt::print("time = {}\n", std::chrono::duration<double>(end -
  // start).count());

  // Pivoted QR to determine sampling nodes
  auto kmatt = nda::matrix<dcomplex, F_layout>(transpose(kmat));
  auto start = std::chrono::high_resolution_clock::now();
  auto piv = nda::zeros<int>(niom_dense * niom_dense);
  auto tau = nda::vector<dcomplex>(3 * r * r + r);
  nda::lapack::geqp3(kmatt, piv, tau);

  // Estimate rank
  int niom_skel = 0;
  for (int k = 0; k < 3 * r * r + r; ++k) {
    if (abs(kmatt(k, k)) < eps) {
      niom_skel = k;
      break;
    }
  }

  // Extract skeleton nodes from pivots
  auto dlr2d_if = nda::array<int, 2>(niom_skel, 2);
  for (int k = 0; k < niom_skel; ++k) {
    auto [n1, n2] = ind2sub(piv(k), niom_dense);
    dlr2d_if(k, 0) = n1 - niom_dense / 2;
    dlr2d_if(k, 1) = n2 - niom_dense / 2;
  }

  // Write dlr2d_if to hdf5 file
  h5::file file(path + filename, 'w');
  h5::group mygroup(file);
  h5::write(mygroup, "dlr2d_if", dlr2d_if);

  fmt::print("DLR rank squared = {}\n", r * r);
  fmt::print("System matrix rank = {}\n\n", niom_skel);
}

nda::array<int, 2> read_dlr2d_if(std::string path, std::string filename) {
  h5::file file(path + filename, 'r');
  h5::group mygroup(file);
  auto dlr2d_if = h5::read<nda::array<int, 2>>(mygroup, "dlr2d_if");
  return dlr2d_if;
}

std::tuple<nda::array<int, 2>, nda::array<int, 2>>
read_dlr2d_rfif(std::string path, std::string filename) {
  h5::file file(path + filename, 'r');
  h5::group mygroup(file);
  auto dlr2d_rfidx = h5::read<nda::array<int, 2>>(mygroup, "dlr2d_rfidx");
  auto dlr2d_if = h5::read<nda::array<int, 2>>(mygroup, "dlr2d_if");
  return {dlr2d_rfidx, dlr2d_if};
}

// Obtain 2D DLR nodes using reduced fine grid, mixed fermionic/bosonic
// representation
void get_dlr2d_if_reduced(nda::vector<double> dlr_rf,
                          nda::vector<int> dlr_if_fer,
                          nda::vector<int> dlr_if_bos, double eps,
                          std::string path, std::string filename) {

  int rankmethod = 1;

  int r = dlr_rf.size();

  // Get fine 2D Matsubara frequency sampling grid
  auto nu2didx = nda::array<int, 2>(3 * r * r + r, 2);
  for (int m = 0; m < r; ++m) {
    for (int n = 0; n < r; ++n) {
      nu2didx(m * r + n, 0) = dlr_if_fer(m); // nu1 = (2*m_j + 1)*i*pi
      nu2didx(m * r + n, 1) = dlr_if_fer(n); // nu2 = (2*n_j + 1)*i*pi

      nu2didx(r * r + m * r + n, 0) =
          dlr_if_bos(n) - dlr_if_fer(m) -
          1; // nu1 = 2*n_k*i*pi - (2*m_j+1)*i*pi = (2*(n_k-m_j-1)+1)*i*pi
      nu2didx(r * r + m * r + n, 1) = dlr_if_fer(m); // nu2 = (2*m_j + 1)*i*pi

      nu2didx(2 * r * r + m * r + n, 0) =
          dlr_if_fer(m); // nu1 = (2*m_j + 1)*i*pi
      nu2didx(2 * r * r + m * r + n, 1) =
          dlr_if_bos(n) - dlr_if_fer(m) -
          1; // nu2 = 2*n_k*i*pi - (2*m_j+1)*i*pi = (2*(n_k-m_j-1)+1)*i*pi
    }

    nu2didx(3 * r * r + m, 0) = dlr_if_fer(m); // nu1 = (2*m_j + 1)*i*pi
    nu2didx(3 * r * r + m, 1) =
        -dlr_if_fer(m) - 1; // nu2 = -(2*m_j + 1)*i*pi = (2*(-m_j-1)+1)*i*pi
  }

  auto nu2d = (2 * nu2didx + 1) * pi * 1i;

  // Get system matrix for dense grid
  auto kmat = nda::matrix<dcomplex, F_layout>(3 * r * r + r, 3 * r * r + r);

  // Regular part
  for (int k = 0; k < r; ++k) {
    for (int l = 0; l < r; ++l) {
      for (int n = 0; n < 3 * r * r + r; ++n) {

        kmat(n, k * r + l) = k_if(nu2didx(n, 0), dlr_rf(k), Fermion) *
                             k_if(nu2didx(n, 1), dlr_rf(l), Fermion);
        // kmat(n, r * r + k * r + l) =
        //     k_if(nu2didx(n, 1), dlr_rf(k), Fermion) *
        //     my_k_if_boson(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l));
        // kmat(n, 2 * r * r + k * r + l) =
        //     k_if(nu2didx(n, 0), dlr_rf(k), Fermion) *
        //     my_k_if_boson(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l));
        kmat(n, r * r + k * r + l) =
            k_if(nu2didx(n, 1), dlr_rf(k), Fermion) *
            k_if(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l), Boson);
        kmat(n, 2 * r * r + k * r + l) =
            k_if(nu2didx(n, 0), dlr_rf(k), Fermion) *
            k_if(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l), Boson);

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
    for (int n = 0; n < 3 * r * r + r; ++n) {
      if (nu2didx(n, 0) == -nu2didx(n, 1) - 1) {
        kmat(n, 3 * r * r + k) = k_if(nu2didx(n, 0), dlr_rf(k), Fermion);
      } else {
        kmat(n, 3 * r * r + k) = 0;
      }
      // if (nu2d(n, 0) == -nu2d(n, 1)) {
      //   kmat(n, 3 * r * r + k) = ker(nu2d(n, 0), dlr_rf(k));
      // } else {
      //   kmat(n, 3 * r * r + k) = 0;
      // }
    }
  }

  // Pivoted QR to determine sampling nodes
  auto kmatt = nda::matrix<dcomplex, F_layout>(transpose(kmat));
  auto start = std::chrono::high_resolution_clock::now();
  auto piv = nda::zeros<int>(3 * r * r + r);
  auto tau = nda::vector<dcomplex>(3 * r * r + r);
  nda::lapack::geqp3(kmatt, piv, tau);

  // Estimate rank
  int niom_skel = 0;
  if (rankmethod == 1) {
    for (int k = 0; k < 3 * r * r + r; ++k) {
      if (abs(kmatt(k, k)) < eps) {
        niom_skel = k;
        break;
      }
    }
  } else {
    double errsq = 0;
    for (int k = 3 * r * r + r - 1; k >= 0; --k) {
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
    dlr2d_if(k, 0) = nu2didx(piv(k), 0);
    dlr2d_if(k, 1) = nu2didx(piv(k), 1);
  }

  // Write dlr2d_if to hdf5 file
  h5::file file(path + filename, 'w');
  h5::group mygroup(file);
  h5::write(mygroup, "dlr2d_if", dlr2d_if);

  fmt::print("DLR rank squared = {}\n", r * r);
  fmt::print("System matrix rank = {}\n\n", niom_skel);
}

// Obtain 2D DLR nodes using reduced fine grid, recompression of basis
void get_dlr2d_rfif(nda::vector<double> dlr_rf, nda::vector<int> dlr_if_fer,
                    nda::vector<int> dlr_if_bos, double eps, std::string path,
                    std::string filename) {

  int rankmethod = 1;

  int r = dlr_rf.size();

  // Get fine 2D Matsubara frequency sampling grid
  auto nu2didx = nda::array<int, 2>(3 * r * r + r, 2);
  for (int m = 0; m < r; ++m) {
    for (int n = 0; n < r; ++n) {
      nu2didx(m * r + n, 0) = dlr_if_fer(m); // nu1 = (2*m_j + 1)*i*pi
      nu2didx(m * r + n, 1) = dlr_if_fer(n); // nu2 = (2*n_j + 1)*i*pi

      nu2didx(r * r + m * r + n, 0) =
          dlr_if_bos(n) - dlr_if_fer(m) -
          1; // nu1 = 2*n_k*i*pi - (2*m_j+1)*i*pi = (2*(n_k-m_j-1)+1)*i*pi
      nu2didx(r * r + m * r + n, 1) = dlr_if_fer(m); // nu2 = (2*m_j + 1)*i*pi

      nu2didx(2 * r * r + m * r + n, 0) =
          dlr_if_fer(m); // nu1 = (2*m_j + 1)*i*pi
      nu2didx(2 * r * r + m * r + n, 1) =
          dlr_if_bos(n) - dlr_if_fer(m) -
          1; // nu2 = 2*n_k*i*pi - (2*m_j+1)*i*pi = (2*(n_k-m_j-1)+1)*i*pi
    }

    nu2didx(3 * r * r + m, 0) = dlr_if_fer(m); // nu1 = (2*m_j + 1)*i*pi
    nu2didx(3 * r * r + m, 1) =
        -dlr_if_fer(m) - 1; // nu2 = -(2*m_j + 1)*i*pi = (2*(-m_j-1)+1)*i*pi
  }

  auto nu2d = (2 * nu2didx + 1) * pi * 1i;

  // Get system matrix for dense grid
  auto kmat = nda::matrix<dcomplex, F_layout>(3 * r * r + r, 3 * r * r + r);

  // Regular part
  for (int k = 0; k < r; ++k) {
    for (int l = 0; l < r; ++l) {
      for (int n = 0; n < 3 * r * r + r; ++n) {

        kmat(n, k * r + l) = k_if(nu2didx(n, 0), dlr_rf(k), Fermion) *
                             k_if(nu2didx(n, 1), dlr_rf(l), Fermion);
        kmat(n, r * r + k * r + l) =
            k_if(nu2didx(n, 1), dlr_rf(k), Fermion) *
            my_k_if_boson(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l));
        kmat(n, 2 * r * r + k * r + l) =
            k_if(nu2didx(n, 0), dlr_rf(k), Fermion) *
            my_k_if_boson(nu2didx(n, 0) + nu2didx(n, 1) + 1, dlr_rf(l));

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
    for (int n = 0; n < 3 * r * r + r; ++n) {
      if (nu2didx(n, 0) == -nu2didx(n, 1) - 1) {
        kmat(n, 3 * r * r + k) = k_if(nu2didx(n, 0), dlr_rf(k), Fermion);
      } else {
        kmat(n, 3 * r * r + k) = 0;
      }
      // if (nu2d(n, 0) == -nu2d(n, 1)) {
      //   kmat(n, 3 * r * r + k) = ker(nu2d(n, 0), dlr_rf(k));
      // } else {
      //   kmat(n, 3 * r * r + k) = 0;
      // }
    }
  }

  // Pivoted QR to determine basis
  auto kmat_copy = nda::matrix<dcomplex, F_layout>(kmat);
  auto piv = nda::zeros<int>(3 * r * r + r);
  auto tau = nda::vector<dcomplex>(3 * r * r + r);
  nda::lapack::geqp3(kmat, piv, tau);

  // Estimate rank
  int r2d = 0;
  if (rankmethod == 1) {
    for (int k = 0; k < 3 * r * r + r; ++k) {
      if (abs(kmat(k, k)) < eps) {
        r2d = k;
        break;
      }
    }
  } else {
    double errsq = 0;
    for (int k = 3 * r * r + r - 1; k >= 0; --k) {
      errsq += pow(abs(kmat(k, k)), 2);
      if (sqrt(errsq) > eps) {
        r2d = k;
        break;
      }
    }
  }

  // Extract basis functions from pivots
  auto dlr2d_rfidx = nda::array<int, 2>(r2d, 4);
  int idx = 0;
  double k = 0, l = 0;
  for (int i = 0; i < r2d; ++i) {
    idx = piv(i);
    if (idx < r * r) {
      std::tie(k, l) = ind2sub_c(idx, r);
      dlr2d_rfidx(i, 0) = 0;
      dlr2d_rfidx(i, 1) = k;
      dlr2d_rfidx(i, 2) = l;
    } else if (idx < 2 * r * r) {
      std::tie(k, l) = ind2sub_c(idx - r * r, r);
      dlr2d_rfidx(i, 0) = 1;
      dlr2d_rfidx(i, 1) = k;
      dlr2d_rfidx(i, 2) = l;
    } else if (idx < 3 * r * r) {
      std::tie(k, l) = ind2sub_c(idx - 2 * r * r, r);
      dlr2d_rfidx(i, 0) = 2;
      dlr2d_rfidx(i, 1) = k;
      dlr2d_rfidx(i, 2) = l;
    } else {
      dlr2d_rfidx(i, 0) = 3;
      dlr2d_rfidx(i, 1) = idx - 3 * r * r;
    }
  }

  auto kmat2 = nda::matrix<dcomplex, F_layout>(r2d, 3 * r * r + r);
  for (int k = 0; k < r2d; ++k) {
    kmat2(k, _) = kmat_copy(_, piv(k));
  }
  piv = 0;
  auto tau2 = nda::vector<dcomplex>(r2d);
  nda::lapack::geqp3(kmat2, piv, tau);

  // Extract skeleton nodes from pivots
  auto dlr2d_if = nda::array<int, 2>(r2d, 2);
  for (int k = 0; k < r2d; ++k) {
    dlr2d_if(k, 0) = nu2didx(piv(k), 0);
    dlr2d_if(k, 1) = nu2didx(piv(k), 1);
  }

  // Write data to hdf5 file
  h5::file file(path + filename, 'w');
  h5::group mygroup(file);
  h5::write(mygroup, "dlr2d_rfidx", dlr2d_rfidx);
  h5::write(mygroup, "dlr2d_if", dlr2d_if);

  fmt::print("DLR rank squared = {}\n", r * r);
  fmt::print("System matrix rank = {}\n\n", r2d);
}

nda::matrix<dcomplex, F_layout>
get_kmat(double beta, nda::vector<double> dlr_rf, nda::array<int, 2> dlr2d_if) {

  int r = dlr_rf.size();
  int niom_skel = dlr2d_if.shape(0);

  // Get system matrix for dense grid
  auto kmat = nda::matrix<dcomplex, F_layout>(niom_skel, 3 * r * r + r);
  // std::complex<double> nu1 = 0, nu2 = 0;

  // Regular part
  for (int k = 0; k < r; ++k) {
    for (int l = 0; l < r; ++l) {
      for (int n = 0; n < niom_skel; ++n) {

        kmat(n, k * r + l) = beta * beta *
                             k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion) *
                             k_if(dlr2d_if(n, 1), dlr_rf(l), Fermion);
        // kmat(n, r * r + k * r + l) =
        //     beta * beta * k_if(dlr2d_if(n, 1), dlr_rf(k), Fermion) *
        //     my_k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l));
        // kmat(n, 2 * r * r + k * r + l) =
        //     beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion) *
        //     my_k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l));
        kmat(n, r * r + k * r + l) =
            beta * beta * k_if(dlr2d_if(n, 1), dlr_rf(k), Fermion) *
            k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l));
        kmat(n, 2 * r * r + k * r + l) =
            beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion) *
            k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, dlr_rf(l));

        // nu1 = (2 * dlr2d_if(n, 0) + 1) * pi * 1i;
        // nu2 = (2 * dlr2d_if(n, 1) + 1) * pi * 1i;
        // kmat(n, k * r + l) =
        //     beta * beta * ker(nu1, dlr_rf(k)) * ker(nu2, dlr_rf(l));
        // kmat(n, r * r + k * r + l) =
        //     beta * beta * ker(nu2, dlr_rf(k)) * ker(nu1 + nu2, dlr_rf(l));
        // kmat(n, 2 * r * r + k * r + l) =
        //     beta * beta * ker(nu1, dlr_rf(k)) * ker(nu1 + nu2, dlr_rf(l));
      }
    }
  }

  // Singular part
  for (int k = 0; k < r; ++k) {
    for (int n = 0; n < niom_skel; ++n) {
      // nu1 = (2 * dlr2d_if(n, 0) + 1) * pi * 1i;
      if (dlr2d_if(n, 0) == -dlr2d_if(n, 1) - 1) {
        kmat(n, 3 * r * r + k) =
            beta * beta * k_if(dlr2d_if(n, 0), dlr_rf(k), Fermion);
        // kmat(n, 3 * r * r + k) = beta * ker(nu1, dlr_rf(k));
      } else {
        kmat(n, 3 * r * r + k) = 0;
      }
    }
  }

  return kmat;
}

nda::matrix<dcomplex, F_layout>
get_kmat_compressed(double beta, nda::vector<double> dlr_rf,
                    nda::array<int, 2> dlr2d_rfidx,
                    nda::array<int, 2> dlr2d_if) {

  int r = dlr_rf.size();
  int r2d = dlr2d_if.shape(0);

  // Get system matrix for dense grid
  auto kmat = nda::matrix<dcomplex, F_layout>(r2d, r2d);

  // Regular part
  double omk = 0, oml = 0;
  for (int i = 0; i < r2d; ++i) {
    omk = dlr_rf(dlr2d_rfidx(i, 1));
    oml = dlr_rf(dlr2d_rfidx(i, 2));
    for (int n = 0; n < r2d; ++n) {
      if (dlr2d_rfidx(i, 0) == 0) {
        kmat(n, i) = beta * beta * k_if(dlr2d_if(n, 0), omk, Fermion) *
                     k_if(dlr2d_if(n, 1), oml, Fermion);
      } else if (dlr2d_rfidx(i, 0) == 1) {
        kmat(n, i) = beta * beta * k_if(dlr2d_if(n, 1), omk, Fermion) *
                     my_k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, oml);
      } else if (dlr2d_rfidx(i, 0) == 2) {
        kmat(n, i) = beta * beta * k_if(dlr2d_if(n, 0), omk, Fermion) *
                     my_k_if_boson(dlr2d_if(n, 0) + dlr2d_if(n, 1) + 1, oml);
      } else {
        kmat(n, i) = beta * k_if(dlr2d_if(n, 0), omk, Fermion);
      }
    }
  }

  return kmat;
}

nda::array<dcomplex, 1>
dlr2d_vals2coefs_compressed(nda::matrix<dcomplex, F_layout> kmat,
                            nda::vector_const_view<dcomplex> vals) {

  int r2d = vals.size();
  auto coef = nda::array<dcomplex, 1>(r2d);
  coef = vals;

  auto s = nda::vector<double>(r2d); // Singular values (not needed)
  int rank = 0;                      // Rank (not needed)
  nda::lapack::gelss(kmat, coef, s, 0.0, rank);

  return coef;
}

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>>
dlr2d_vals2coefs(nda::matrix<dcomplex, F_layout> kmat,
                 nda::vector_const_view<dcomplex> vals, int r) {

  int m = vals.size();
  int n = 3 * r * r + r;
  auto tmp = nda::array<dcomplex, 1>(n);
  tmp(nda::range(m)) = vals;

  auto s = nda::vector<double>(m); // Singular values (not needed)
  int rank = 0;                    // Rank (not needed)
  nda::lapack::gelss(kmat, tmp, s, 0.0, rank);

  // auto u = nda::matrix<dcomplex, F_layout>(m, m);
  // auto vt = nda::matrix<dcomplex, F_layout>(n, n);
  // auto s = nda::vector<double>(m, n);
  // nda::lapack::gesvd(kmat, s, u, vt);
  // auto smat = nda::matrix<double, F_layout>(m, n);
  // smat = 0;
  // for (int k = 0; k < m; ++k) {
  //   smat(k, k) = 1.0/s(k);
  //   if (s(k) < 2.22e-16) {PRINT(s(k));}
  // }
  // tmp = conj(transpose(vt)) * (smat * (conj(transpose(u)) * vals));

  auto coefreg = nda::array<dcomplex, 3>(3, r, r);
  auto coefsng = nda::array<dcomplex, 1>(r);
  reshape(coefreg, 3 * r * r) = tmp(nda::range(3 * r * r));
  coefsng = tmp(nda::range(3 * r * r, 3 * r * r + r));

  return {coefreg, coefsng};
}

std::tuple<nda::array<dcomplex, 4>, nda::array<dcomplex, 2>>
dlr2d_vals2coefs_many(nda::matrix<dcomplex, F_layout> kmat,
                      nda::array_const_view<dcomplex, 2, F_layout> vals,
                      int r) {

  int m = vals.shape(0);
  int nrhs = vals.shape(1);
  int n = 3 * r * r + r;
  auto tmp = nda::array<dcomplex, 2, F_layout>(n, nrhs);
  tmp(nda::range(m), _) = vals;

  auto s = nda::vector<double>(m); // Singular values (not needed)
  int rank = 0;                    // Rank (not needed)
  nda::lapack::gelss(kmat, tmp, s, 0.0, rank);

  auto coefreg = nda::array<dcomplex, 4>(nrhs, 3, r, r);
  auto coefsng = nda::array<dcomplex, 2>(nrhs, r);

  for (int j = 0; j < nrhs; ++j) {
    reshape(coefreg(j, _, _, _), 3 * r * r) = tmp(nda::range(3 * r * r), j);
    coefsng(j, _) = tmp(nda::range(3 * r * r, 3 * r * r + r), j);
  }

  return {coefreg, coefsng};
}

// Evaluate 2D DLR expansion
// Channel = 1 for particle-particle, = 2 for particle-hole
std::complex<double>
dlr2d_coefs2eval(double beta, nda::vector<double> dlr_rf,
                 nda::array_const_view<dcomplex, 3> gc,
                 nda::array_const_view<dcomplex, 1> gc_sing, int m, int n,
                 int channel) {

  int r = dlr_rf.size(); // # DLR basis functions

  // Make sure coefficient array is 3xrxr
  if (gc.shape(0) != 3)
    throw std::runtime_error("First dim of coefficient array must be 3.");
  if ((gc.shape(1) != r) || (gc.shape(2) != r))
    throw std::runtime_error("Second and third dims of coefficient array must "
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
  auto kb = nda::vector<dcomplex>(r);
  // auto kfmn = nda::vector<dcomplex>(r);
  for (int k = 0; k < r; ++k) {
    kfm(k) = k_if(mm, dlr_rf(k), Fermion);
    kfn(k) = k_if(n, dlr_rf(k), Fermion);
    // kb(k) = my_k_if_boson(mm + n + 1, dlr_rf(k));
    kb(k) = k_if_boson(mm + n + 1, dlr_rf(k));
    // kfmn(k) = ker(2*(mm + n + 1)*1i*pi, dlr_rf(k));
  }

  // Evaluate DLR expansion
  auto g = beta * beta *
           (nda::blas::dot(kfm, matvecmul(gc(0, _, _), kfn)) +
            nda::blas::dot(kfn, matvecmul(gc(1, _, _), kb)) +
            nda::blas::dot(kfm, matvecmul(gc(2, _, _), kb)));

  if (mm + n + 1 == 0) {
    g += beta * beta * nda::blas::dot(gc_sing, kfm);
  }

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

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>>
uncompress_basis(int r, nda::array<int, 2> dlr2d_rfidx,
                 nda::array<dcomplex, 1> coef) {
  int r2d = dlr2d_rfidx.shape(0);

  auto coefreg = nda::zeros<dcomplex>(3, r, r);
  auto coefsng = nda::zeros<dcomplex>(r);

  int idx = 0, j = 0, k = 0, l = 0;
  for (int i = 0; i < r2d; ++i) {
    if (dlr2d_rfidx(i, 0) < 3) {
      coefreg(dlr2d_rfidx(i, 0), dlr2d_rfidx(i, 1), dlr2d_rfidx(i, 2)) =
          coef(i);
    } else {
      coefsng(dlr2d_rfidx(i, 1)) = coef(i);
    }
  }

  return {coefreg, coefsng};
}

std::string get_filename(double lambda, double eps, int niom_dense) {

  std::ostringstream filenameStream;
  filenameStream << "dlr2d_if_" << lambda << "_" << std::scientific
                 << std::setprecision(2) << eps << "_" << niom_dense << ".h5";
  return filenameStream.str();
}

std::string get_filename(double lambda, double eps, bool compressed) {

  std::ostringstream filenameStream;
  if (!compressed) {
    filenameStream << "dlr2d_if_reduced_" << lambda << "_" << std::scientific
                 << std::setprecision(2) << eps << ".h5";
  } else {
    filenameStream << "dlr2d_if_reduced_" << lambda << "_" << std::scientific
                 << std::setprecision(2) << eps << "_compressed" << ".h5";
  }
  return filenameStream.str();
}

// Estimate rank of a square matrix A for which the full pivoted QR
// decomposition has been obtained using the function geqp3. The
// upper-triangular matrix R, which is used to estimate the rank, is stored in
// the upper-triangular part of A.
//
// We use Eqn. (4.3) from Halko, Martinsson, Tropp, SIAM Rev. 2011 to obtain an
// efficient randomized algorithm to estimate the rank in a manner which
// guarantees (with very high probability) that the spectral norm error of the
// resulting estimate of A is less than eps. The failure probability is
// alpha^(-nvec), and is determined by the "paranoia factor" alpha > 1, and the
// number of random vectors nvec used in the algorithm. The total work is
// proportional to nvec. Larger values of alpha lead to a less optimal estimate
// of the rank, so the most optimal solution is obtained by choosing alpha close
// to 1 and a correspondingly large value of nvec.
int estimate_rank(nda::matrix_const_view<dcomplex, F_layout> a, double eps,
                  double alpha, int nvec) {
  int n = a.shape(0);

  // Set up random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> d(0.0, 1.0);

  // Generate random Gaussian vectors
  auto x = nda::matrix<dcomplex>(n, nvec);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < nvec; ++j) {
      x(i, j) = d(gen) + 1i * d(gen);
    }
  }

  // Extract upper triangular matrix R
  auto r = nda::matrix<dcomplex, F_layout>(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      r(i, j) = a(i, j);
    }
  }

  // Multiply R by random vectors
  auto y = r * x;

  // Compute cumulative l2 norms of columns of x, starting from the bottom
  auto xnorm = nda::zeros<double>(x.shape());
  for (int j = 0; j < nvec; ++j) {
    // Compute cumulative sum of squares of elements in column j, starting from
    // bottom
    xnorm(n - 1, j) = pow(abs(x(n - 1, j)), 2);
    for (int i = n - 2; i >= 0; --i) {
      xnorm(i, j) = pow(abs(x(i, j)), 2) + xnorm(i + 1, j);
    }
  }

  // Compute cumulative l2 norms of columns of y, starting from the bottom
  auto ynorm = nda::zeros<double>(y.shape());
  for (int j = 0; j < nvec; ++j) {
    // Compute cumulative sum of squares of elements in column j, starting from
    // bottom
    ynorm(n - 1, j) = pow(abs(y(n - 1, j)), 2);
    for (int i = n - 2; i >= 0; --i) {
      ynorm(i, j) = pow(abs(y(i, j)), 2) + ynorm(i + 1, j);
    }
  }

  ynorm /= xnorm;

  // Take maximum of cumulative l2 norms over random vectors
  for (int i = 0; i < n; ++i) {
    ynorm(i, 0) = max_element(ynorm(i, _));
  }

  // Estimate rank
  int rank = 0;
  double epssc = eps / (alpha * sqrt(2 / pi));
  for (int i = 0; i < n - 1; ++i) {
    if (ynorm(i + 1, 0) < epssc * epssc) {
      rank = i + 1;
      break;
    }
  }

  return rank;
}

// Get bosonic DLR Matsubara frequency grid with modified kernel
nda::vector<int> get_dlr_if_boson(double lambda,
                                  nda::vector_const_view<double> dlr_rf) {
  int nmax = fineparams(lambda).nmax;
  int r = dlr_rf.size();
  auto dlr_if_boson = nda::vector<int>(r);

  auto kmat = nda::matrix<dcomplex>(2 * nmax + 1, r);

  for (int n = -nmax; n <= nmax; ++n) {
    for (int j = 0; j < r; ++j) {
      kmat(nmax + n, j) = my_k_if_boson(n, dlr_rf(j));
    }
  }

  auto [q, norms, piv] = pivrgs(kmat, 1e-100);
  std::sort(piv.begin(), piv.end()); // Sort pivots in ascending order
  for (int i = 0; i < r; ++i) {
    dlr_if_boson(i) = piv(i) - nmax;
  }

  return dlr_if_boson;
}