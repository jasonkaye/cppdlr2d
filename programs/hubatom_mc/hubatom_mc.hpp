/*!
 *\file hubatom_mc.hpp
 *\brief Hubbard atom fit to CT-INT Matsubara frequency data
 */

#include <cppdlr2d/cppdlr2d.hpp>

/**
 * @brief Compare CT-INT data and DLR expansions for multiple data files.
 *
 * For each data file, computes DLR expansions, errors vs reference, and outputs
 * all errors.
 *
 * @param beta Inverse temperature
 * @param u Hubbard interaction
 * @param lambda DLR cutoff
 * @param eps DLR error tolerance
 * @param nmaxtst # Matsubara freqs per dim in test grid
 * @param compressgrid Use compressed DLR grid
 * @param compressbasis Use compressed DLR basis
 * @param datafiles Vector of CT-INT HDF5 file paths
 */
void hubatom_mc_compare(double beta, double u, double lambda, double eps,
                        int nmaxtst, bool compressgrid, bool compressbasis,
                        const std::vector<std::string> &datafiles,
                        bool output = false);

/**
 * @brief Load CT-INT data from file and extract chi_s, chi_d, chi_m arrays,
 * plus metadata.
 * @param datafile Path to CT-INT HDF5 file
 * @return Tuple of chi_s, chi_d, chi_m arrays and n_cycles, num_threads
 * (integers)
 */
std::tuple<nda::array<std::complex<double>, 2>,
           nda::array<std::complex<double>, 2>,
           nda::array<std::complex<double>, 2>, int, int>
load_chi_data(const std::string &datafile);

/**
 * @brief Fit CT-INT data to DLR and evaluate DLR expansion on a test grid.
 *
 * Reads CT-INT data, subsamples on the DLR grid, computes DLR coefficients, and
 * evaluates the DLR expansion on a test grid. Returns original chi_s, chi_d,
 * chi_m data as well as the corresponding DLR expansions on the test grid.
 *
 * @param beta Inverse temperature
 * @param lambda DLR cutoff
 * @param eps DLR error tolerance
 * @param nmaxtst # Matsubara freqs per dim in test grid
 * @param compressgrid Use compressed DLR grid
 * @param compressbasis Use compressed DLR basis
 * @param datafile Path to CT-INT HDF5 file
 * @return Tuple of chi_s, chi_d, chi_m arrays (data grid), chi_s, chi_d, chi_m
 * arrays (test grid), n_cycles, num_threads
 */
std::tuple<
    nda::array<std::complex<double>, 2>, nda::array<std::complex<double>, 2>,
    nda::array<std::complex<double>, 2>, nda::array<std::complex<double>, 2>,
    nda::array<std::complex<double>, 2>, nda::array<std::complex<double>, 2>,
    int, int>
get_dlr_from_ctint_data(double beta, double lambda, double eps, int nmaxtst,
                        bool compressgrid, bool compressbasis,
                        const std::string &datafile);

/**
 * @brief Compute exact reference chi_s, chi_d, chi_m arrays on a data grid.
 * @param nmax Matsubara freq grid cutoff index
 * @param beta Inverse temperature
 * @param u Hubbard interaction
 * @return Tuple of chi_s, chi_d, chi_m arrays (nda::array<std::complex<double>,
 * 2>)
 */
std::tuple<nda::array<std::complex<double>, 2>,
           nda::array<std::complex<double>, 2>,
           nda::array<std::complex<double>, 2>>
get_ref_data(int nmax, double beta, double u);
