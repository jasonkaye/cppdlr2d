#pragma once
#include "cppdlr/cppdlr.hpp"
#include "nda/nda.hpp"

#include <numbers>
#include <string>

namespace dlr2d {

  using namespace cppdlr;
  using std::numbers::pi;

  using fmatrix            = nda::matrix<dcomplex, nda::F_layout>;
  using fmatrix_const_view = nda::matrix_const_view<dcomplex, nda::F_layout>;

  /*!
 * \brief Get standard filename used by \ref build_dlr2d_if_fullgrid to store 2D
 * Matsubara frequency DLR grid
 *
 * \param[in] lambda      DLR cutoff
 * \param[in] eps         Error tolerance
 * \param[in] niom_dense  # Matsubara frequencies per dimension in fine grid
 *
 * \return Standard filename describing grid parameters
 */
std::string get_filename(double lambda, double eps, int niom_dense);

/*!
 * \brief Get standard filename used by \ref build_dlr2d_if and \ref
 * build_dlr2d_ifrf to store 2D Matsubara frequency DLR grid
 *
 * \param[in] lambda      DLR cutoff
 * \param[in] eps         Error tolerance
 * \param[in] compressed  (=false (default) for use with \ref build_dlr2d_if,
 * =true for use with \ref build_dlr2d_ifrf)
 *
 * \return Standard filename describing grid parameters
 */
std::string get_filename(double lambda, double eps, bool compressed = false);

/*!
 * \brief Get standard filename used by \ref build_dlr2d_if_3term to store 2D
 * Matsubara frequency DLR grid
 *
 * \param[in] lambda      DLR cutoff
 * \param[in] eps         Error tolerance
 *
 * \return Standard filename describing grid parameters
 */
std::string get_filename_3term(double lambda, double eps);

/*!
 * \brief Estimate rank of a square matrix from its full pivoted QR
 * decomposition
 *
 * Assumes QR decomposition was computed using LAPACK geqp3. The
 * upper-triangular matrix R, which is used to estimate the rank, is stored in
 * the upper-triangular part of A.

 * We use Eqn. (4.3) from Halko, Martinsson, Tropp, SIAM Rev. 2011 to obtain an
 * efficient randomized algorithm to estimate the rank in a manner which
 * guarantees (with very high probability) that the spectral norm error of the
 * resulting estimate of A is less than eps. The failure probability is
 * alpha^(-nvec), and is determined by the "paranoia factor" alpha > 1, and the
 * number of random vectors nvec used in the algorithm. The total work is
 * proportional to nvec. Larger values of alpha lead to a less optimal estimate
 * of the rank, so the most optimal solution is obtained by choosing alpha close
 * to 1 and a correspondingly large value of nvec.
 *
 * \param[in] a     Result of geqp3 on A, containing upper-triangular matrix R
 * \param[in] eps   Error tolerance for rank estimation
 * \param[in] alpha Paranoia factor
 * \param[in] nvec  # random vectors used in algorithm
 *
 * \return Estimated rank of matrix A
 *
 * \note DESPITE SUPPOSED GUARANTEES, THIS FUNCTION HAS SO FAR YIELDED MIXED
 * RESULTS IN LIMITED TESTING, AND SHOULD BE USED WITH CAUTION UNTIL FURTHER
 * TESTING IS DONE.
 */
int estimate_rank(fmatrix_const_view a, double eps,
                  double alpha, int nvec);

/*!
 * \brief Simple definition of imaginary frequency analytic continuation kernel
 *
 * \param[in] nu  Imaginary frequency
 * \param[in] om  Real frequency
 *
 * \return Kernel value
 *
 * \note TODO: This function is old and should be removed.
 */
dcomplex ker(dcomplex nu, double om);

/*!
 * \brief Alternative definition of bosonic imaginary frequency kernel
 *
 * \param[in] n   Imaginary frequency index
 * \param[in] om  Real frequency
 *
 * \return Kernel value
 *
 * \note TODO: This function is old and should probably be removed.
 */
dcomplex my_k_if_boson(int n, double om);

/*!
 * \brief Obtain bosonic 1D DLR Matsubara frequencies using alternative
 * definition of bosonic kernel
 *
 * \param[in] lambda  DLR cutoff
 * \param[in] dlr_rf      1D DLR real frequencies
 *
 * \return Bosonic 1D DLR Matsubara frequencies
 *
 * \note TODO: I don't remember the purpose of this function and it should
 * probably be removed eventually.
 */
nda::vector<int> get_dlr_if_boson(double lambda,
                                  nda::vector_const_view<double> dlr_rf);

/*!
 * \brief Convert linear index of nxn column-major array to index pair
 * (zero-indexed)
 *
 * \param[in] idx Linear index
 * \param[in] n   Array dimension
 *
 * \return Index pair (zero-indexed)
 */
std::tuple<int, int> ind2sub(int idx, int n);

/*!
 * \brief Convert linear index of nxn row-major array to index pair
 * (zero-indexed)
 *
 * \param[in] idx Linear index
 * \param[in] n   Array dimension
 *
 * \return Index pair (zero-indexed)
 */
std::tuple<int, int> ind2sub_c(int idx, int n);

} // namespace dlr2d
