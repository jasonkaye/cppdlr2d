#pragma once
#include "cppdlr/cppdlr.hpp"
#include "nda/nda.hpp"

namespace dlr2d {

/*!
 * \brief Obtain 2D DLR Matsubara frequency grid
 *
 * This function generates an HDF5 file in the specified path containing the
 * 2D DLR Matsubara frequency grid points in terms of Matsubara frequency index
 * pairs.
 *
 * It uses the method proposed in Kiese et al., "Discrete Lehmann representation
 * of three-point functions", arXiv:2405.06716, which involves building the fine
 * Matsubara frequency grid from combinations of 1D DLR grid points.
 *
 * \param[in] dlr_rf      1D DLR real frequencies
 * \param[in] dlr_if_fer  Fermionic 1D DLR Matsubara frequencies
 * \param[in] dlr_if_bos  Bosonic 1D DLR Matsubara frequencies
 * \param[in] eps         Error tolerance
 * \param[in] path        Path to directory in which to save 2D DLR Mat. freqs.
 * \param[in] filename    Name of file in which to save 2D DLR Mat. freqs.
 *
 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 */
void build_dlr2d_if(nda::vector<double> dlr_rf, nda::vector<int> dlr_if_fer,
                    nda::vector<int> dlr_if_bos, double eps, std::string path,
                    std::string filename);

/*!
 * \brief Obtain 2D DLR Matsubara frequency grid using three-term DLR
 *
 * This function generates an HDF5 file in the specified path containing the
 * 2D DLR Matsubara frequency grid points in terms of Matsubara frequency index
 * pairs.
 *
 * It differs from the method used in \ref build_dlr2d_if in that it uses a
 * Lehmann representation of only three terms, rather than four, obtained by
 * absorbing one term into the others.
 *
 * \param[in] dlr_rf      1D DLR real frequencies
 * \param[in] dlr_if_fer  Fermionic 1D DLR Matsubara frequencies
 * \param[in] dlr_if_bos  Bosonic 1D DLR Matsubara frequencies
 * \param[in] eps         Error tolerance
 * \param[in] path        Path to directory in which to save 2D DLR Mat. freqs.
 * \param[in] filename    Name of file in which to save 2D DLR Mat. freqs.
 *
 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 */
void build_dlr2d_if_3term(nda::vector<double> dlr_rf,
                          nda::vector<int> dlr_if_fer,
                          nda::vector<int> dlr_if_bos, double eps,
                          std::string path, std::string filename);

/*!
 * \brief Obtain 2D DLR Matsubara frequency grid and compressed 2D DLR real
 * frequency grid
 *
 * This function generates an HDF5 file in the specified path containing the
 * 2D DLR Matsubara frequency grid points in terms of Matsubara frequency index
 * pairs and the 2D DLR real frequency grid points in terms of 1D DLR real
 * frequency grid index pairs.
 *
 * To generate the 2D DLR Matsubara frequency grid, it uses the same method as
 * in \ref build_dlr2d_if. It then compresses the 2D DLR real frequency grid
 * using the pivoted Gram-Schmidt algorithm on the columns of the row-compressed
 * kernel matrix.
 *
 * \param[in] dlr_rf      1D DLR real frequencies
 * \param[in] dlr_if_fer  Fermionic 1D DLR Matsubara frequencies
 * \param[in] dlr_if_bos  Bosonic 1D DLR Matsubara frequencies
 * \param[in] eps         Error tolerance
 * \param[in] path        Path to directory in which to save 2D DLR Mat. freqs.
 * \param[in] filename    Name of file in which to save 2D DLR Mat. freqs.
 *
 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 *
 * \note THIS FUNCTION IS OLD AND MAY NOT WORK PROPERLY. TEST BEFORE USING.
 *
 * \note This is not the method proposed in Kiese et al., "Discrete Lehmann
 * representation of three-point functions", arXiv:2405.06716, which does not
 * recompress the 2D DLR real frequency grid. That method yields an
 * "overcomplete" representation, whereas this method yields a fully compressed
 * representation.
 */
void build_dlr2d_ifrf(nda::vector<double> dlr_rf, nda::vector<int> dlr_if_fer,
                      nda::vector<int> dlr_if_bos, double eps, std::string path,
                      std::string filename);

/*!
 * \brief Obtain 2D DLR Matsubara frequency grid, using all Matsubara
 * frequencies up to specified cutoff as fine grid
 *
 * This function generates an HDF5 file in the specified path containing the
 * 2D DLR Matsubara frequency grid points in terms of Matsubara frequency index
 * pairs.
 *
 * \param[in] dlr_rf      1D DLR real frequencies
 * \param[in] niom_dense  # Matsubara frequencies per dimension in fine grid
 * \param[in] eps         Error tolerance
 * \param[in] path        Path to directory in which to save 2D DLR Mat. freqs.
 * \param[in] filename    Name of file in which to save 2D DLR Mat. freqs.
 *
 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 *
 * \note THIS FUNCTION IS OLD AND MAY NOT WORK PROPERLY. TEST BEFORE USING.
 *
 * \note This is not the standard way of generating the 2D DLR grid, and will be
 * slow for large values of the DLR cutoff Lambda and tolerance epsilon. The
 * function build_dlr2d_if is the standard, more efficient approach, and
 * corresponds to the method proposed in Kiese et al., "Discrete Lehmann
 * representation of three-point functions", arXiv:2405.06716.
 */
void build_dlr2d_if_fullgrid(nda::vector<double> dlr_rf, int niom_dense,
                             double eps, std::string path,
                             std::string filename);

/*!
 * \brief Read 2D DLR Matsubara frequency grid from file
 *
 * This functions reads the 2D DLR Matsubara frequency grid from an HDF5 file
 * produced using one of the following functions: \ref build_dlr2d_if, \ref
 * build_dlr2d_if_3term, \ref build_dlr2d_if_fullgrid.
 *
 * \param[in] path     Path to directory containing 2D DLR Mat. freqs.
 * \param[in] filename Name of file containing 2D DLR Mat. freqs.
 *
 * \return 2D DLR Matsubara frequency grid as an array containing Mat. freq.
 * index pairs.
 *
 * \note See the documentation for the functions noted above for more
 * information on how the grid is produced, and its format.
 */
nda::array<int, 2> read_dlr2d_if(std::string path, std::string filename);

/*!
 * \brief Read 2D DLR Matsubara frequency grid and compressed 2D DLR real
 * frequency grid from file
 *
 * This functions reads the 2D DLR Matsubara frequency grid and the compressed
 * 2D DLR real frequency grid from an HDF5 file produced using the function \ref
 * build_dlr2d_ifrf.
 *
 * \param[in] path     Path to directory containing 2D DLR Mat. freqs.
 * \param[in] filename Name of file containing 2D DLR Mat. freqs.
 *
 * \return 2D DLR Matsubara frequency grid as an array containing Matsubara
 * frequency index pairs, and compressed 2D DLR real frequency grid as an array
 * containing real frequency index pairs.
 *
 * \note See the documentation for the functions noted above for more
 * information on how the grids is produced, and their formats.
 */
std::tuple<nda::array<int, 2>, nda::array<int, 2>>
read_dlr2d_rfif(std::string path, std::string filename);

/*!
 * \brief Build matrix which maps coefficients of a 2D DLR expansion to its
 * values on the 2D DLR imaginary (Matsubara) frequency grid
 *
 * \param[in] beta      Inverse temperature
 * \param[in] dlr_rf    1D DLR real frequencies
 * \param[in] dlr2d_if  2D DLR imaginary frequency grid
 *
 * \return Coefficients to values matrix
 */
nda::matrix<dcomplex, F_layout> build_cf2if(double beta,
                                            nda::vector<double> dlr_rf,
                                            nda::array<int, 2> dlr2d_if);

/*!
 * \brief Build matrix which maps coefficients of a 2D DLR expansion to its
 * values on the 2D DLR imaginary (Matsubara) frequency grid, using three-term
 * DLR
 *
 * This function differs from \ref build_cf2if in that it uses a Lehmann
 * representation of only three terms, rather than four, obtained by absorbing
 * one term into the others.
 *
 * \param[in] beta      Inverse temperature
 * \param[in] dlr_rf    1D DLR real frequencies
 * \param[in] dlr2d_if  2D DLR Matsubara frequency grid
 *
 * \return Coefficients to values matrix
 */
nda::matrix<dcomplex, F_layout> build_cf2if_3term(double beta,
                                                  nda::vector<double> dlr_rf,
                                                  nda::array<int, 2> dlr2d_if);

/*!
 * \brief Build matrix which maps coefficients of a 2D DLR expansion to its
 * values on the 2D DLR imaginary (Matsubara) frequency grid, using a
 * recompressed 2D DLR
 *
 * This function differs from \ref build_cf2if in that it uses a DLR
 * with recompressed 2D real frequency pairs; see \ref build_dlr2d_ifrf and \ref
 * read_dlr2d_rfif.
 *
 * \param[in] beta          Inverse temperature
 * \param[in] dlr_rf        1D DLR real frequencies
 * \param[in] dlr2d_rfidx   Compressed 2D DLR real frequency index pairs
 * \param[in] dlr2d_if      2D DLR Matsubara frequency grid
 *
 * \return Coefficients to values matrix
 */
nda::matrix<dcomplex, F_layout>
build_cf2if_square(double beta, nda::vector<double> dlr_rf,
                   nda::array<int, 2> dlr2d_rfidx, nda::array<int, 2> dlr2d_if);

/*!
 * \brief Transform values of a 2D DLR expansion on the 2D DLR imaginary
 * (Matsubara) frequency grid to its coefficients
 *
 * \param[in] cf2if  Coefficients to values matrix
 * \param[in] vals   Values of 2D DLR expansion on 2D DLR Mat. freq. grid
 * \param[in] r      # basis functions in 1D DLR
 *
 * \return Coefficients of 2D DLR expansion
 *
 * \note The matrix \p cf2if should be obtained using \ref build_cf2if.
 */
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>>
vals2coefs_if(nda::matrix<dcomplex, F_layout> cf2if,
              nda::vector_const_view<dcomplex> vals, int r);

/*!
 * \brief Transform values of multiple 2D DLR expansions on the 2D DLR imaginary
 * (Matsubara) frequency grid to their coefficients
 *
 * \param[in] cf2if  Coefficients to values matrix
 * \param[in] vals   Values of 2D DLR expansions on 2D DLR Mat. freq. grid
 * \param[in] r      # basis functions in 1D DLR
 *
 * \return Coefficients of 2D DLR expansions
 *
 * \note The matrix \p cf2if should be obtained using \ref build_cf2if.
 *
 * \note TODO: Replace \ref vals2coefs_if with a properly templated version of
 * this function.
 */
std::tuple<nda::array<dcomplex, 4>, nda::array<dcomplex, 2>>
vals2coefs_if_many(nda::matrix<dcomplex, F_layout> cf2if,
                   nda::array_const_view<dcomplex, 2, F_layout> vals, int r);

/*!
 * \brief Transform values of a 2D DLR expansion on the 2D DLR imaginary
 * (Matsubara) frequency grid to its coefficients, using three-term DLR
 *
 * This function differs from \ref vals2coefs_if in that it uses a Lehmann
 * representation of only three terms, rather than four, obtained by absorbing
 * one term into the others.
 *
 * \param[in] cf2if  Coefficients to values matrix
 * \param[in] vals   Values of 2D DLR expansion on 2D DLR Mat. freq. grid
 * \param[in] r      # basis functions in 1D DLR
 *
 * \return Coefficients of 2D DLR expansion
 *
 * \note The matrix \p cf2if should be obtained using \ref build_cf2if_3term.
 */
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>>
vals2coefs_if_3term(nda::matrix<dcomplex, F_layout> cf2if,
                    nda::vector_const_view<dcomplex> vals, int r);

/*!
 * \brief Transform values of multiple 2D DLR expansions on the 2D DLR imaginary
 * (Matsubara) frequency grid to their coefficients, using three-term DLR
 *
 * This function differs from \ref vals2coefs_if_many in that it uses a Lehmann
 * representation of only three terms, rather than four, obtained by absorbing
 * one term into the others.
 *
 * \param[in] cf2if  Coefficients to values matrix
 * \param[in] vals   Values of 2D DLR expansions on 2D DLR Mat. freq. grid
 * \param[in] r      # basis functions in 1D DLR
 *
 * \return Coefficients of 2D DLR expansions
 *
 * \note The matrix \p cf2if should be obtained using \ref build_cf2if_3term.
 */
std::tuple<nda::array<dcomplex, 4>, nda::array<dcomplex, 2>>
vals2coefs_if_many_3term(nda::matrix<dcomplex, F_layout> cf2if,
                         nda::array_const_view<dcomplex, 2, F_layout> vals,
                         int r);

/*!
 * \brief Transform values of a 2D DLR expansion on the 2D DLR imaginary
 * (Matsubara) frequency grid to its coefficients, using a recompressed 2D DLR
 *
 * This function differs from \ref vals2coefs_if in that it uses a DLR
 * with recompressed 2D real frequency pairs; see \ref build_dlr2d_ifrf and \ref
 * read_dlr2d_rfif.
 *
 * \param[in] cf2if  Coefficients to values matrix
 * \param[in] vals   Values of 2D DLR expansion on 2D DLR Mat. freq. grid
 *
 * \return Coefficients of 2D DLR expansion
 *
 * \note The matrix \p cf2if should be obtained using \ref build_cf2if_square.
 */
nda::array<dcomplex, 1>
vals2coefs_if_square(nda::matrix<dcomplex, F_layout> cf2if,
                     nda::vector_const_view<dcomplex> vals);

// Evaluate 2D DLR expansion
std::complex<double> coefs2eval_if(double beta, nda::vector<double> dlr_rf,
                                   nda::array_const_view<dcomplex, 3> gc,
                                   nda::array_const_view<dcomplex, 1> gc_skel,
                                   int m, int n, int channel);

std::complex<double>
coefs2eval_if_3term(double beta, nda::vector<double> dlr_rf,
                    nda::array_const_view<dcomplex, 3> gc,
                    nda::array_const_view<dcomplex, 1> gc_sing, int m, int n,
                    int channel);

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>>
uncompress_basis(int r, nda::array<int, 2> dlr2d_rfidx,
                 nda::array<dcomplex, 1> coef);

// Generate name for dlr2d_if file
std::string get_filename(double lambda, double eps, int niom_dense);

// Generate name for dlr2d_if file
std::string get_filename(double lambda, double eps, bool compressed = 0);

// Generate name for dlr2d_if file with 2 terms
std::string get_filename_two_terms(double lambda, double eps);

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

// Imaginary frequency kernel function
std::complex<double> ker(std::complex<double> nu, double om);

// Function to convert linear index of nxn column-major array to index pair
// (zero-indexed)
std::tuple<int, int> ind2sub(int idx, int n);

} // namespace dlr2d