#pragma once

#include "utils.hpp"

namespace cppdlr2d {

  /*!
 * \brief Obtain 2D DLR "product" Matsubara frequency grid
 *
 * This function builds a grid composed of a union of products of 1D DLR grids,
 * as described by Eq. 21 of Kiese et al. ["Discrete Lehmann representation of
 * three-point functions", PRB (2025)]. This is the "fine" grid from which we
 * typically select the 2D DLR grid points. We note that here, the grid points
 * corresponding to the fourth row of Eq. 21 (the contribution corresponding to
 * singular terms) are omitted, as these points are redundant with the other
 * terms as long as the zero bosonic frequency is included in the 1D DLR grid.
 *
 * \param[in] lambda   DLR cutoff parameter
 * \param[in] dlr_rf   1D DLR real frequency grid
 *
 * \return Grid points returned as an array of Matsubara frequency index pairs.
 *
 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n).
 */
  nda::array<int, 2> build_prod_if(double lambda, nda::vector_const_view<double> dlr_rf);

  /*!
 * \copydoc build_prod_if(double, nda::vector_const_view<double>)
 *
 * Rather than returning the grid, this overload writes the grid index pairs to
 *  an HDF5 file.
 *
 * \param[in] path        Path to directory in which to save index pairs
 * \param[in] filename    Name of file in which to save index pairs
 */
  void build_prod_if(double lambda, nda::vector_const_view<double> dlr_rf, const std::string &path, const std::string &filename);

  /*!
 * \brief Obtain 2D DLR Matsubara frequency grid and real frequency grids
 *
 * This function generates the 2D DLR Matsubara frequency grid points in terms
 * of Matsubara frequency pairs, as well as the real frequency pairs determining
 * the 2D DLR basis functions. Both grids are represented as arrays of index
 * pairs.
 *
 * If compressgrid is false, the imaginary frequency grid is taken to be the
 * "product" grid produced by \ref build_prod_if. If true, this grid is
 * compressed using pivoted QR.
 *
 * If compressbasis is false, the real frequency grid is taken to be a product
 * of 1D DLR real frequency grids. If true, this grid is compressed using
 * pivoted QR.
 *
 * The method proposed in Kiese et al. ["Discrete Lehmann representation of
 * three-point functions", PRB (2025)] uses compressgrid=true and
 * compressbasis=false.
 *
 * \param[in] lambda        DLR cutoff parameter
 * \param[in] eps           Error tolerance
 * \param[in] compressgrid  Compress grid using pivoted QR? (default: true)
 * \param[in] compressbasis Compress basis using pivoted QR? (default: false)
 *
 * \return Tuple containing 2D DLR real and imaginary (respectively) frequency
 * grids as arrays of index pairs.
 *
 * \note For a fermionic Matsubara frequency i*nu_n = (2n+1)*pi/beta, we refer
 * to n as its index. An index pair (m, n) corresponds to the 2D Matsubara
 * frequency point (i nu_m, i nu_n). For a real frequency pair, its indices
 * refer to that of the corresponding 1D DLR real frequency grid point.
 */
  std::tuple<nda::array<int, 2>, nda::array<int, 2>> build_dlr2d(double lambda, double eps, bool compressgrid = true, bool compressbasis = false);

  /*!
 * \copydoc build_dlr2d(double, double, bool, bool)
 *
 * Rather than returning the grids, this overload writes the 2D DLR Matsubara
 * frequency index pairs to an HDF5 file.
 *
 * \param[in] path        Path to directory in which to save index pairs
 * \param[in] filename    Name of file in which to save index pairs
 */
  void build_dlr2d(double lambda, double eps, const std::string &path, const std::string &filename, bool compressgrid = true,
                   bool compressbasis = false);

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
  void build_dlr2d_if_3term(double lambda, double eps, std::string path, std::string filename);

  nda::array<int, 2> build_dlr2d_if_3term(double lambda, double eps);

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
  std::tuple<nda::array<int, 2>, nda::array<int, 2>> read_dlr2d(std::string path, std::string filename);

  /*!
 * \brief Build matrix which maps coefficients of a 2D DLR expansion to its
 * values on a 2D imaginary (Matsubara) frequency grid
 *
 * \param[in] beta      Inverse temperature
 * \param[in] dlr_rf    1D DLR real frequencies
 * \param[in] dlr2d_if  2D imaginary frequency grid indices
 *
 * \return Coefficients to values matrix
 */
  fmatrix build_cf2if(double beta, nda::vector_const_view<double> dlr_rf, nda::array_const_view<int, 2> dlr2d_if);

  /*!
 *  \copydoc build_cf2if(double, nda::vector_const_view<double>,
 * nda::array_const_view<int, 2>)
 *
 * Allows specifying 2D DLR real frequencies as subset of full product grid.
 *
 * \param[in] dlr2d_rf  2D DLR real frequency grid indices
 */
  fmatrix build_cf2if(double beta, nda::vector_const_view<double> dlr_rf, nda::array_const_view<int, 2> dlr2d_if,
                      nda::array_const_view<int, 2> dlr2d_rf);

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
 * \brief Transform imaginary frequency values of a 2D DLR expansion to
 * coefficients
 *
 * \param[in] r         # basis functions in 1D DLR
 * \param[in] cf2if     Coefficients to values matrix
 * \param[in] vals      Values of 2D DLR expansion on imag. freq. grid
 * \param[in] dlr2d_rf  Real frequency grid indices
 *
 * \return Coefficients of 2D DLR expansion
 *
 * \note The matrix \p cf2if should be obtained using \ref build_cf2if.
 */
  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>> vals2coefs(int r, fmatrix cf2if, nda::vector_const_view<dcomplex> vals,
                                                                          nda::array_const_view<int, 2> dlr2d_rf);

  /*!
 * \brief Transform imaginary frequency values of a 2D DLR expansion to
 * coefficients, multiple expansions
 *
 * \param[in] r         # basis functions in 1D DLR
 * \param[in] cf2if     Coefficients to values matrix
 * \param[in] vals      Values of 2D DLR expansion on imag. freq. grid
 * \param[in] dlr2d_rf  Real frequency grid indices
 *
 * \return Coefficients of 2D DLR expansions
 *
 * \note The matrix \p cf2if should be obtained using \ref build_cf2if.
 */
  std::tuple<nda::array<dcomplex, 4>, nda::array<dcomplex, 2>>
  vals2coefs_many(int r, fmatrix cf2if, nda::array_const_view<dcomplex, 2, nda::F_layout> vals, nda::array_const_view<int, 2> dlr2d_rf);

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
  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 1>> vals2coefs_if_3term(fmatrix cf2if, nda::vector_const_view<dcomplex> vals, int r);

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
  std::complex<double> coefs2eval_if(double beta, nda::vector<double> dlr_rf, nda::array_const_view<dcomplex, 3> gc_reg,
                                     nda::array_const_view<dcomplex, 1> gc_sng, int m, int n, int channel);

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
  std::complex<double> coefs2eval_if_3term(double beta, nda::vector<double> dlr_rf, nda::array_const_view<dcomplex, 3> gc_reg,
                                           nda::array_const_view<dcomplex, 1> gc_sng, int m, int n, int channel);

} // namespace cppdlr2d