#pragma once
#include "cppdlr/cppdlr.hpp"
#include "nda/nda.hpp"

#include <numbers>
#include <string>

namespace dlr2d {

  using dcomplex = std::complex<double>;

  /**
 * @brief Compute the l2 norm of a 2D complex array, normalized by beta^2.
 * @param arr Complex, 2D array
 * @param beta Inverse temperature normalization
 * @return l2 norm (double)
 */
  template <nda::ArrayOfRank<2> A>
    requires std::is_same_v<std::complex<double>, nda::get_value_t<A>>
  double l2_norm(const A &arr, double beta) {
    return sqrt(sum(pow(abs(arr), 2))) / (beta * beta);
  }

  /**
 * @brief Compute the linf norm (max abs) of a 2D complex array.
 * @param arr Complex, 2D array
 * @return linf norm (double)
 */
  template <nda::ArrayOfRank<2> A>
    requires std::is_same_v<std::complex<double>, nda::get_value_t<A>>
  double linf_norm(const A &arr) {
    return max_element(abs(arr));
  }

  /**
 * @brief Compute the DLR nodes for the particle-hole channel.
 *
 * Given a 2D DLR index array for the particle-particle channel, returns the
 * corresponding 2D DLR index array for the particle-hole channel.
 *
 * @param dlr2d_if 2D DLR indices
 * @return Particle-hole channel DLR indices
 */
  nda::array<int, 2> get_dlr2d_if_ph(nda::array_const_view<int, 2> dlr2d_if);

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
 * \param[in] lambda        DLR cutoff
 * \param[in] eps           Error tolerance
 * \param[in] compressgrid  Compress grid using pivoted QR? (default: true)
 * \param[in] compressbasis Compress basis using pivoted QR? (default: false)
 *
 * \return Standard filename describing grid parameters
 */
  std::string get_filename(double lambda, double eps, bool compressgrid, bool compressbasis);

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
 *
 * Method 1 estimates based on the values of the diagonal elements of R.
 *
 * Method 2 estimates based on the sum of the squares of the lower right entries
 * of R.
 *
 * Method 3 uses Eqn. (4.3) from Halko, Martinsson, Tropp, SIAM Rev. 2011 to
 * obtain an efficient randomized algorithm to estimate the rank in a manner
 * which guarantees (with very high probability) that the spectral norm error of
 * the resulting estimate of A is less than eps. The failure probability is
 * alpha^(-nvec), and is determined by the "paranoia factor" alpha > 1, and the
 * number of random vectors nvec used in the algorithm. The total work is
 * proportional to nvec. Larger values of alpha lead to a less optimal estimate
 * of the rank, so the most optimal solution is obtained by choosing alpha close
 * to 1 and a correspondingly large value of nvec.
 *
 * \param[in] a     Result of geqp3 on A, containing upper-triangular matrix R
 * \param[in] eps   Error tolerance for rank estimation
 * \param[in] method Rank estimation method (1, 2, or 3)
 * \param[in] alpha Paranoia factor (if method = 3)
 * \param[in] nvec  # random vectors used in algorithm (if method = 3)
 *
 * \return Estimated rank of matrix A
 *
 * \note DESPITE SUPPOSED GUARANTEES, METHOD 3 HAS SO FAR YIELDED MIXED
 * RESULTS IN LIMITED TESTING, AND SHOULD BE USED WITH CAUTION UNTIL FURTHER
 * TESTING IS DONE. DO NOT TRUST DEFAULTS.
 */
  int estimate_rank(fmatrix_const_view a, double eps, int method = 1, double alpha = 2.0, int nvec = 100);

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
  nda::vector<int> get_dlr_if_boson(double lambda, nda::vector_const_view<double> dlr_rf);

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
