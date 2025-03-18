#pragma once

#include "utils.hpp"

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
 * \param[in] lambda      DLR cutoff parameter
 * \param[in] eps         Error tolerance
 * \param[in] path        Path to directory in which to save 2D DLR Mat. freqs.
 * \param[in] filename    Name of file in which to save 2D DLR Mat. freqs.
 *
 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 */
void build_dlr2d_if(double lambda, double eps, std::string path,
                    std::string filename);

nda::array<int, 2> build_dlr2d_if(double lambda, double eps);

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
 * \param[in] lambda      DLR cutoff parameter
 * \param[in] eps         Error tolerance
 * \param[in] path        Path to directory in which to save 2D DLR Mat. freqs.
 * \param[in] filename    Name of file in which to save 2D DLR Mat. freqs.
 *
 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 */
void build_dlr2d_if_3term(double lambda, double eps, std::string path,
                          std::string filename);

nda::array<int, 2> build_dlr2d_if_3term(double lambda, double eps);

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
 * \param[in] lambda      DLR cutoff parameter
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
void build_dlr2d_ifrf(double lambda, double eps, std::string path,
                      std::string filename);

std::pair<nda::array<int, 2>, nda::array<int, 2>> build_dlr2d_ifrf(double lambda, double eps);

/*!
 * \brief Obtain 2D DLR Matsubara frequency grid, using all Matsubara
 * frequencies up to specified cutoff as fine grid
 *
 * This function generates an HDF5 file in the specified path containing the
 * 2D DLR Matsubara frequency grid points in terms of Matsubara frequency index
 * pairs.
 *
 * \param[in] lambda      DLR cutoff parameter
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
void build_dlr2d_if_fullgrid(double lambda, int niom_dense, double eps,
                             std::string path, std::string filename);

nda::array<int, 2> build_dlr2d_if_fullgrid(double lambda, int niom_dense, double eps);

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
fmatrix build_cf2if(double beta, nda::vector<double> dlr_rf, nda::array<int, 2> dlr2d_if);

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
fmatrix build_cf2if_3term(double beta, nda::vector<double> dlr_rf, nda::array<int, 2> dlr2d_if);

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
fmatrix build_cf2if_square(double beta, nda::vector<double> dlr_rf, nda::array<int, 2> dlr2d_rfidx, nda::array<int, 2> dlr2d_if);

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
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>> vals2coefs_if(fmatrix cf2if, nda::vector_const_view<dcomplex> vals,
                                                                           int r);

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
vals2coefs_if_many(fmatrix cf2if, nda::array_const_view<dcomplex, 2, nda::F_layout> vals, int r);

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
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>> vals2coefs_if_3term(fmatrix cf2if,
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
vals2coefs_if_many_3term(fmatrix cf2if, nda::array_const_view<dcomplex, 2, nda::F_layout> vals, int r);

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
nda::array<dcomplex, 1> vals2coefs_if_square(fmatrix cf2if, nda::vector_const_view<dcomplex> vals);

/*!
 * \brief Evaluate a 2D DLR expansion at a given fermionic/fermionic Matsubara
 * frequency point
 *
 * \param[in] beta    Inverse temperature
 * \param[in] dlr_rf  1D DLR real frequencies
 * \param[in] gc_reg  2D DLR regular expansion coefficients
 * \param[in] gc_sng  1D DLR singular expansion coefficients
 * \param[in] m       First index of Matsubara frequency point
 * \param[in] n       Second index of Matsubara frequency point
 * \param[in] channel Channel index (=1 for particle-particle, =2 for
 particle-hole)

 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 */
std::complex<double> coefs2eval_if(double beta, nda::vector<double> dlr_rf,
                                   nda::array_const_view<dcomplex, 3> gc_reg,
                                   nda::array_const_view<dcomplex, 1> gc_sng,
                                   int m, int n, int channel);

/*!
 * \brief Evaluate a 2D DLR expansion at a given fermionic/fermionic Matsubara
 * frequency point, using three-term DLR
 *
 * This function differs from \ref coefs2eval_if in that it uses a Lehmann
 * representation of only three terms, rather than four, obtained by absorbing
 * one term into the others.
 *
 * \param[in] beta    Inverse temperature
 * \param[in] dlr_rf  1D DLR real frequencies
 * \param[in] gc_reg  2D DLR regular expansion coefficients
 * \param[in] gc_sng  1D DLR singular expansion coefficients
 * \param[in] m       First index of Matsubara frequency point
 * \param[in] n       Second index of Matsubara frequency point
 * \param[in] channel Channel index (=1 for particle-particle, =2 for
 particle-hole)

 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 */
std::complex<double>
coefs2eval_if_3term(double beta, nda::vector<double> dlr_rf,
                    nda::array_const_view<dcomplex, 3> gc_reg,
                    nda::array_const_view<dcomplex, 1> gc_sng, int m, int n,
                    int channel);

/*!
 * \brief Convert compressed 2D DLR expansion coefficients to ordinary
 * (overcomplete) 2D DLR expansion coefficient storage format
 *
 * This function must be called before using \ref coefs2eval_if to evaluate a
 * compressed DLR expansion with DLR frequency pairs obtained using \ref
 * build_dlr2d_ifrf. It returns 2D DLR expansion coefficients with zeros
 * corresponding to DLR frequency pairs which were not selected during the
 * compression.
 *
 * \param[in] r             # basis functions in 1D DLR
 * \param[in] dlr2d_rfidx   Compressed 2D DLR real frequency index pairs
 * \param[in] gc            Compressed 2D DLR expansion coefficients
 *
 * \return 2D DLR regular and singular expansion coefficients
 */
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>>
uncompress_basis(int r, nda::array<int, 2> dlr2d_rfidx,
                 nda::array<dcomplex, 1> gc);

} // namespace dlr2d
