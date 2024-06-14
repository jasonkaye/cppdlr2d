#pragma once
#include "cppdlr/cppdlr.hpp"
#include "nda/nda.hpp"

// Imaginary frequency kernel function
std::complex<double> ker(std::complex<double> nu, double om);

// Function to convert linear index of nxn column-major array to index pair
// (zero-indexed)
std::tuple<int, int> ind2sub(int idx, int n);

// Obtain 2D DLR nodes
void get_dlr2d_if(nda::vector<double> dlr_rf, int niom_dense, double eps,
                  std::string path, std::string filename);

// Obtain 2D DLR nodes using reduced fine grid, mixed fermionic/bosonic
// representation
void get_dlr2d_if_reduced(nda::vector<double> dlr_rf,
                          nda::vector<int> dlr_if_fer,
                          nda::vector<int> dlr_if_bos, double eps,
                          std::string path, std::string filename);

void get_dlr2d_rfif(nda::vector<double> dlr_rf, nda::vector<int> dlr_if_fer,
                    nda::vector<int> dlr_if_bos, double eps, std::string path,
                    std::string filename);

// Read 2D DLR nodes from hdf5 file
nda::array<int, 2> read_dlr2d_if(std::string path, std::string filename);

std::tuple<nda::array<int, 2>, nda::array<int, 2>>
read_dlr2d_rfif(std::string path, std::string filename);

// Obtain kernel matrix for fitting 2D DLR expansion
nda::matrix<dcomplex, F_layout>
get_kmat(double beta, nda::vector<double> dlr_rf, nda::array<int, 2> dlr2d_if);

nda::matrix<dcomplex, F_layout>
get_kmat_compressed(double beta, nda::vector<double> dlr_rf,
                    nda::array<int, 2> dlr2d_rfidx,
                    nda::array<int, 2> dlr2d_if);

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>>
dlr2d_vals2coefs(nda::matrix<dcomplex, F_layout> kmat,
                 nda::vector_const_view<dcomplex> vals, int r);

std::tuple<nda::array<dcomplex, 4>, nda::array<dcomplex, 2>>
dlr2d_vals2coefs_many(nda::matrix<dcomplex, F_layout> kmat,
                      nda::array_const_view<dcomplex, 2, F_layout> vals, int r);

nda::array<dcomplex, 1>
dlr2d_vals2coefs_compressed(nda::matrix<dcomplex, F_layout> kmat,
                            nda::vector_const_view<dcomplex> vals);

// Evaluate 2D DLR expansion
std::complex<double>
dlr2d_coefs2eval(double beta, nda::vector<double> dlr_rf,
                 nda::array_const_view<dcomplex, 3> gc,
                 nda::array_const_view<dcomplex, 1> gc_skel, int m, int n,
                 int channel);

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>>
uncompress_basis(int r, nda::array<int, 2> dlr2d_rfidx,
                 nda::array<dcomplex, 1> coef);

// Generate name for dlr2d_if file
std::string get_filename(double lambda, double eps, int niom_dense);

// Generate name for dlr2d_if file
std::string get_filename(double lambda, double eps);

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
                  double alpha, int nvec);

nda::vector<int> get_dlr_if_boson(double lambda,
                                  nda::vector_const_view<double> dlr_rf);