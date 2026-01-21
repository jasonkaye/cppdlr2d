#pragma once

#include "dlr2d.hpp"

namespace dlr2d {

  // Compute polarization by convolution-based algorithm
  nda::vector<dcomplex> polarization(double beta, double lambda, double eps, cppdlr::imtime_ops const &itops, cppdlr::imfreq_ops const &ifops_fer,
                                     cppdlr::imfreq_ops const &ifops_bos, nda::array_const_view<dcomplex, 1> fc,
                                     nda::array_const_view<dcomplex, 1> gc, nda::array_const_view<dcomplex, 3> lambc,
                                     nda::array_const_view<dcomplex, 1> lambc_sing);

  // Compute polarization by convolution-based algorithm using 3 term Lehman representation
  nda::vector<dcomplex> polarization_3term(double beta, double lambda, double eps, cppdlr::imtime_ops const &itops,
                                           cppdlr::imfreq_ops const &ifops_fer, cppdlr::imfreq_ops const &ifops_bos,
                                           nda::array_const_view<dcomplex, 1> fc, nda::array_const_view<dcomplex, 1> gc,
                                           nda::array_const_view<dcomplex, 3> lambc, nda::array_const_view<dcomplex, 1> lambc_sing);

  // Compute polarization by residue calculus-based algorithm
  nda::vector<dcomplex> polarization_res(double beta, cppdlr::imfreq_ops const &ifops_fer, cppdlr::imfreq_ops const &ifops_bos,
                                         nda::array_const_view<dcomplex, 1> fc, nda::array_const_view<dcomplex, 1> gc,
                                         nda::array_const_view<dcomplex, 3> lambc, nda::array_const_view<dcomplex, 1> lambc_sing);

  // Compute contribution to polarization of constant part of vertex, assumed to
  // be 1
  nda::vector<dcomplex> polarization_const(double beta, cppdlr::imtime_ops const &itops, cppdlr::imfreq_ops const &ifops_bos,
                                           nda::array_const_view<dcomplex, 1> fc, nda::array_const_view<dcomplex, 1> gc);

} // namespace dlr2d
