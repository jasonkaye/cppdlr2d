/*!
 *\file siam.hpp
 *\brief Header file for single-impurity Anderson model example
 */

#include "../../src/dlr2d.hpp"
#include "../../src/polarization.hpp"

using namespace dlr2d;

/*!
 * \brief Driver function for single-impurity Anderson model example, all
 * functions
 *
 * This function expands the Green's function, three-point correlators, and
 * vertex functions in all channels in the DLR, for the SIAM model example
 * described in Kiese et al., "Discrete Lehmann representation of three-point
 * functions", arXiv:2405.06716. It then computes the polarization function in
 * all channels. It measures the error of all of these representations against
 * data computed using exact diagonalization which must be supplied externally.
 *
 * \param[in] beta          Inverse temperature
 * \param[in] u             Hubbard interaction
 * \param[in] lambda        DLR cutoff
 * \param[in] eps           Error tolerance
 * \param[in] niomtst       # Matsubara freqs per dim in test grid
 * \param[in] nbos_tst      # bosonic Matsubara freqs in test grid for
 * polarization
 * \param[in] reduced       Use reduced fine Matsubara freq grid
 * \param[in] compressbasis Overcomplete or compressed DLR basis
 * \param[in] niom_dense    # Matsubara freqs in fine grid (only used if
 * reduced=false)
 *
 * \return Vector containing problem parameters and errors, for analysis and
 * plotting
 */
nda::vector<double> siam_allfuncs(double beta, double u, double lambda,
                                  double eps, int niomtst, int nbos_tst,
                                  bool reduced, bool compressbasis,
                                  int niom_dense = 0);

/*!
 * \brief Driver function for single-impurity Anderson model example, all
 * functions, using three-term 2D DLR
 *
 * This function expands the Green's function, three-point correlators, and
 * vertex functions in all channels in the DLR, for the SIAM model example
 * described in Kiese et al., "Discrete Lehmann representation of three-point
 * functions", arXiv:2405.06716. It then computes the polarization function in
 * all channels. It measures the error of all of these representations against
 * data computed using exact diagonalization which must be supplied externally.
 *
 * It uses the 2D DLR with two regular terms and one singular term, rather than
 * the original representation with three regular terms and one singular term.
 *
 * \param[in] beta          Inverse temperature
 * \param[in] u             Hubbard interaction
 * \param[in] lambda        DLR cutoff
 * \param[in] eps           Error tolerance
 * \param[in] niomtst       # Matsubara freqs per dim in test grid
 * \param[in] nbos_tst      # bosonic Matsubara freqs in test grid for
 * polarization
 *
 * \return Vector containing problem parameters and errors, for analysis and
 * plotting
 */
nda::vector<double> siam_allfuncs_3term(double beta, double u, double lambda, double eps, int niomtst, int nbos_tst);
