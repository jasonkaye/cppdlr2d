#include "polarization.hpp"

namespace dlr2d {

nda::vector<dcomplex> polarization(
    double beta, double lambda, double eps, cppdlr::imtime_ops const &itops,
    cppdlr::imfreq_ops const &ifops_fer, cppdlr::imfreq_ops const &ifops_bos,
    nda::array_const_view<dcomplex, 1> fc,
    nda::array_const_view<dcomplex, 1> gc,
    nda::array_const_view<dcomplex, 3> lambc,
    nda::array_const_view<dcomplex, 1> lambc_sing) {

  auto dlr_rf = ifops_bos.get_rfnodes();
  auto dlr_if_fer = ifops_fer.get_ifnodes();
  auto dlr_if_bos = ifops_bos.get_ifnodes();
  int r = dlr_rf.size();

  // Get finer DLR tau discretization
  double lambda2 = 2 * lambda;
  auto dlr_rf2   = cppdlr::build_dlr_rf(lambda2, eps);
  int r2 = dlr_rf2.size();
  auto itops2    = cppdlr::imtime_ops(lambda2, dlr_rf2);

  // Get coefs -> fine tau vals matrix
  auto cf2itfine = cppdlr::build_k_it(itops2.get_itnodes(), dlr_rf);

  // Get fine coefs -> bosonic imag freq vals matrix
  auto cffine2if = nda::matrix<dcomplex>(r, r2);
  for (int k = 0; k < r2; ++k) {
    for (int j = 0; j < r; ++j) {
      cffine2if(j, k) = k_if(dlr_if_bos(j), dlr_rf2(k), Boson);
    }
  }

  auto pol = nda::zeros<dcomplex>(r);

  // Compute 1/(i Omega_m - omega_k)
  auto kkif = nda::array<dcomplex, 2>(r, r);
  auto iom = 2 * dlr_if_bos * pi * 1i;
  for (int j = 0; j < r; ++j) {
    for (int k = 0; k < r; ++k) {
      // kkif(j, k) = beta / (iom(j) - dlr_rf(k));
      kkif(j, k) = beta * k_if_boson(dlr_if_bos(j), dlr_rf(k));
    }
  }

  // Compute F(tau) and G(tau)
  // auto fit = itops.coefs2vals(fc);
  // auto git = itops.coefs2vals(gc);
  auto fit = matvecmul(cf2itfine, fc);
  auto git = matvecmul(cf2itfine, gc);

  // Compute F_k(i nu_n) = F(i nu_n)/(i nu_n - omega_k) and G_k(i nu_n) = G(i
  // nu_n)/(i nu_n - omega_k)

  auto fif = ifops_fer.coefs2vals(beta, fc);
  auto gif = ifops_fer.coefs2vals(beta, gc);

  auto fkif = nda::matrix<dcomplex>(r, r);
  auto gkif = nda::matrix<dcomplex>(r, r);

  auto inu = (2 * dlr_if_fer + 1) * pi * 1i;
  for (int j = 0; j < r; ++j) {
    for (int k = 0; k < r; ++k) {
      fkif(j, k) = beta * fif(j) / (inu(j) - dlr_rf(k));
      gkif(j, k) = beta * gif(j) / (inu(j) - dlr_rf(k));
    }
  }

  // Compute F_k(tau) and G_k(tau)
  // auto fkit = itops.coefs2vals(ifops_fer.vals2coefs(beta, fkif));
  // auto gkit = itops.coefs2vals(ifops_fer.vals2coefs(beta, gkif));
  auto fkit = matmul(cf2itfine, ifops_fer.vals2coefs(beta, fkif));
  auto gkit = matmul(cf2itfine, ifops_fer.vals2coefs(beta, gkif));

  // Compute contribution to polarization from first non-constant term of vertex
  auto tmp = matmul(fkit, lambc(0, _, _));
  auto polit = nda::zeros<dcomplex>(r2);
  for (int j = 0; j < r2; ++j) {
    for (int k = 0; k < r; ++k) {
      polit(j) += gkit(j, k) * tmp(j, k);
    }
  }

  // pol += beta * ifops_bos.coefs2vals(itops.vals2coefs(polit));
  pol += beta * matvecmul(cffine2if, itops2.vals2coefs(polit));

  // Compute contribution to polarization from second and third non-constant
  // terms of vertex

  auto tmp1 = nda::matrix<dcomplex>(r2, r);
  auto tmp2 = nda::matrix<dcomplex>(r2, r);

  for (int j = 0; j < r2; ++j) {
    for (int k = 0; k < r; ++k) {
      tmp1(j, k) = fit(j) * gkit(j, k);
      tmp2(j, k) = git(j) * fkit(j, k);
    }
  }
  auto tmp3 = matmul(tmp1, lambc(1, _, _));
  auto tmp4 = matmul(tmp2, lambc(2, _, _));

  // tmp3 = ifops_bos.coefs2vals(beta, itops.vals2coefs(tmp3));
  // tmp4 = ifops_bos.coefs2vals(beta, itops.vals2coefs(tmp4));
  auto tmp31 = beta * matmul(cffine2if, itops2.vals2coefs(tmp3));
  auto tmp41 = beta * matmul(cffine2if, itops2.vals2coefs(tmp4));
  for (int j = 0; j < r; ++j) {
    for (int k = 0; k < r; ++k) {
      pol(j) += kkif(j, k) * (tmp31(j, k) + tmp41(j, k));
    }
  }

  // Compute contribution to polarization from singular part of vertex
  auto tmp5 = matvecmul(tmp2, lambc_sing);
  auto tmp6 = cffine2if * itops2.vals2coefs(tmp5);

  int m0idx = 0;
  for (int j = 0; j < r; ++j) {
    if (dlr_if_bos(j) == 0) {
      pol(j) += beta * beta * tmp6(j);
      m0idx = j;
      break;
    }
  }

  // Contribution to polarization from constant part of vertex
  auto fgit = make_regular(nda::array_const_view<dcomplex, 1>(fit) *
                           nda::array_const_view<dcomplex, 1>(git));
  // pol += beta * ifops_bos.coefs2vals(itops.vals2coefs(fgit));
  pol += beta * matvecmul(cffine2if, itops2.vals2coefs(fgit));

  return pol;
}

nda::vector<dcomplex> polarization_3term(
    double beta, double lambda, double eps, cppdlr::imtime_ops const &itops,
    cppdlr::imfreq_ops const &ifops_fer, cppdlr::imfreq_ops const &ifops_bos,
    nda::array_const_view<dcomplex, 1> fc,
    nda::array_const_view<dcomplex, 1> gc,
    nda::array_const_view<dcomplex, 3> lambc,
    nda::array_const_view<dcomplex, 1> lambc_sing) {

  auto dlr_rf = ifops_bos.get_rfnodes();
  auto dlr_if_fer = ifops_fer.get_ifnodes();
  auto dlr_if_bos = ifops_bos.get_ifnodes();
  int r = dlr_rf.size();

  // Get finer DLR tau discretization
  double lambda2 = 2 * lambda;
  auto dlr_rf2 = build_dlr_rf(lambda2, eps);
  int r2 = dlr_rf2.size();
  auto itops2 = imtime_ops(lambda2, dlr_rf2);

  // Get coefs -> fine tau vals matrix
  auto cf2itfine = build_k_it(itops2.get_itnodes(), dlr_rf);

  // Get fine coefs -> bosonic imag freq vals matrix
  auto cffine2if = nda::matrix<dcomplex>(r, r2);
  for (int k = 0; k < r2; ++k) {
    for (int j = 0; j < r; ++j) {
      cffine2if(j, k) = k_if(dlr_if_bos(j), dlr_rf2(k), Boson);
    }
  }

  auto pol = nda::zeros<dcomplex>(r);

  // Compute 1/(i Omega_m - omega_k)
  auto kkif = nda::array<dcomplex, 2>(r, r);
  auto iom = 2 * dlr_if_bos * pi * 1i;
  for (int j = 0; j < r; ++j) {
    for (int k = 0; k < r; ++k) {
      // kkif(j, k) = beta / (iom(j) - dlr_rf(k));
      kkif(j, k) = beta * k_if_boson(dlr_if_bos(j), dlr_rf(k));
    }
  }

  // Compute F(tau) and G(tau)
  // auto fit = itops.coefs2vals(fc);
  // auto git = itops.coefs2vals(gc);
  auto fit = matvecmul(cf2itfine, fc);
  auto git = matvecmul(cf2itfine, gc);

  // Compute F_k(i nu_n) = F(i nu_n)/(i nu_n - omega_k) and G_k(i nu_n) = G(i
  // nu_n)/(i nu_n - omega_k)

  auto fif = ifops_fer.coefs2vals(beta, fc);
  auto gif = ifops_fer.coefs2vals(beta, gc);

  auto fkif = nda::matrix<dcomplex>(r, r);
  auto gkif = nda::matrix<dcomplex>(r, r);

  auto inu = (2 * dlr_if_fer + 1) * pi * 1i;
  for (int j = 0; j < r; ++j) {
    for (int k = 0; k < r; ++k) {
      fkif(j, k) = beta * fif(j) / (inu(j) - dlr_rf(k));
      gkif(j, k) = beta * gif(j) / (inu(j) - dlr_rf(k));
    }
  }

  // Compute F_k(tau) and G_k(tau)
  // auto fkit = itops.coefs2vals(ifops_fer.vals2coefs(beta, fkif));
  // auto gkit = itops.coefs2vals(ifops_fer.vals2coefs(beta, gkif));
  auto fkit = matmul(cf2itfine, ifops_fer.vals2coefs(beta, fkif));
  auto gkit = matmul(cf2itfine, ifops_fer.vals2coefs(beta, gkif));

  // Compute contribution to polarization from second and third non-constant
  // terms of vertex

  auto tmp1 = nda::matrix<dcomplex>(r2, r);
  auto tmp2 = nda::matrix<dcomplex>(r2, r);

  for (int j = 0; j < r2; ++j) {
    for (int k = 0; k < r; ++k) {
      tmp1(j, k) = fit(j) * gkit(j, k);
      tmp2(j, k) = git(j) * fkit(j, k);
    }
  }
  auto tmp3 = matmul(tmp1, lambc(0, _, _));
  auto tmp4 = matmul(tmp2, lambc(1, _, _));

  // tmp3 = ifops_bos.coefs2vals(beta, itops.vals2coefs(tmp3));
  // tmp4 = ifops_bos.coefs2vals(beta, itops.vals2coefs(tmp4));
  auto tmp31 = beta * matmul(cffine2if, itops2.vals2coefs(tmp3));
  auto tmp41 = beta * matmul(cffine2if, itops2.vals2coefs(tmp4));
  for (int j = 0; j < r; ++j) {
    for (int k = 0; k < r; ++k) {
      pol(j) += kkif(j, k) * (tmp31(j, k) + tmp41(j, k));
    }
  }

  // Compute contribution to polarization from singular part of vertex
  auto tmp5 = matvecmul(tmp2, lambc_sing);
  auto tmp6 = cffine2if * itops2.vals2coefs(tmp5);

  int m0idx = 0;
  for (int j = 0; j < r; ++j) {
    if (dlr_if_bos(j) == 0) {
      pol(j) += beta * beta * tmp6(j);
      m0idx = j;
      break;
    }
  }

  // Contribution to polarization from constant part of vertex
  auto fgit = make_regular(nda::array_const_view<dcomplex, 1>(fit) *
                           nda::array_const_view<dcomplex, 1>(git));
  // pol += beta * ifops_bos.coefs2vals(itops.vals2coefs(fgit));
  pol += beta * matvecmul(cffine2if, itops2.vals2coefs(fgit));

  return pol;
}


// Compute polarization
nda::vector<dcomplex>
polarization_res(double beta, imfreq_ops const &ifops_fer,
                 imfreq_ops const &ifops_bos,
                 nda::array_const_view<dcomplex, 1> fc,
                 nda::array_const_view<dcomplex, 1> gc,
                 nda::array_const_view<dcomplex, 3> lambc,
                 nda::array_const_view<dcomplex, 1> lambc_sing) {

  auto dlr_rf = ifops_fer.get_rfnodes();
  int r = dlr_rf.size();   // # DLR basis functions
  auto ej = dlr_rf / beta; // Convert to physical units
  auto dlr_if_fer = ifops_fer.get_ifnodes();
  auto dlr_if_bos = ifops_bos.get_ifnodes();

  auto nu_dlr = ((2 * dlr_if_fer + 1) * pi * 1i) / beta;
  auto om_dlr = (2 * dlr_if_bos * pi * 1i) / beta;

  // Prepare some objects

  // H(j,k) = 1/(E_j - E_k) for j/=k, 0 for j=k
  auto hilb = nda::matrix<dcomplex>(r, r);
  for (int j = 0; j < r; ++j) {
    for (int k = 0; k < r; ++k) {
      if (j == k) {
        hilb(j, k) = 0;
      } else {
        hilb(j, k) = 1.0 / (ej(j) - ej(k));
      }
    }
  }

  // Hm(j,k) = 1/(i omega_m - E_j - E_k)
  auto hilbm = nda::array<dcomplex, 3>(r, r, r);
  for (int m = 0; m < r; ++m) {
    for (int j = 0; j < r; ++j) {
      for (int k = 0; k < r; ++k) {
        hilbm(m, j, k) = 1.0 / (om_dlr(m) - ej(j) - ej(k));
      }
    }
  }

  // Hmsq(j,k) = 1/(i omega_m - E_j - E_k)^2
  auto hilbmsq = nda::array<dcomplex, 3>(r, r, r);
  for (int m = 0; m < r; ++m) {
    for (int j = 0; j < r; ++j) {
      for (int k = 0; k < r; ++k) {
        hilbmsq(m, j, k) =
            1.0 / ((om_dlr(m) - ej(j) - ej(k)) * (om_dlr(m) - ej(j) - ej(k)));
      }
    }
  }

  // nf(j)  = 1/(1+exp(-beta*E_j))
  // nfm(j) = 1/(1+exp(beta*E_j))
  // nfpm(j) = beta * exp(-beta*E_j) / (1+exp(-beta*E_j))^2
  // nfpm(j) = beta * exp(beta*E_j) / (1+exp(beta*E_j))^2
  auto nf = nda::array<double, 1>(r);
  auto nfm = nda::array<double, 1>(r);
  auto nfp = nda::array<double, 1>(r);
  auto nfpm = nda::array<double, 1>(r);
  for (int j = 0; j < r; ++j) {
    nf(j) = -k_it(0.0, beta * ej(j));
    nfm(j) = -k_it(0.0, -ej(j), beta);
    nfp(j) = beta * k_it(0.0, ej(j), beta) * k_it(1.0, ej(j), beta);
    nfpm(j) = beta * k_it(0.0, -ej(j), beta) * k_it(1.0, -ej(j), beta);
  }

  // lamb0jl(j,l) = sum_{k/=j} lambda_kl / (E_j - E_k)
  auto lamb0jl = nda::array<dcomplex, 2>(r, r);
  for (int j = 0; j < r; ++j) {
    for (int l = 0; l < r; ++l) {
      lamb0jl(j, l) = 0;
      for (int k = 0; k < r; ++k) {
        if (k != j) {
          lamb0jl(j, l) += lambc(0, k, l) / (ej(j) - ej(k));
        }
      }
    }
  }

  // lamb0mj(m,j) = sum_{l} lambda_jl / (i \omega_m - E_j - E_l)
  auto lamb0mj = nda::array<dcomplex, 2>(r, r);
  for (int m = 0; m < r; ++m) {
    for (int j = 0; j < r; ++j) {
      lamb0mj(m, j) = 0;
      for (int l = 0; l < r; ++l) {
        lamb0mj(m, j) += lambc(0, j, l) / (om_dlr(m) - ej(j) - ej(l));
      }
    }
  }

  // lamb0mjsq(m,j) = sum_{l} lambda_jl / (i \omega_m - E_j - E_l)^2
  auto lamb0mjsq = nda::array<dcomplex, 2>(r, r);
  for (int m = 0; m < r; ++m) {
    for (int j = 0; j < r; ++j) {
      lamb0mjsq(m, j) = 0;
      for (int l = 0; l < r; ++l) {
        lamb0mjsq(m, j) += lambc(0, j, l) / ((om_dlr(m) - ej(j) - ej(l)) *
                                             (om_dlr(m) - ej(j) - ej(l)));
      }
    }
  }

  // lamb0jk(j,k) = sum_{l/=j} lambda_kl / (E_j - E_l)
  auto lamb0jk = nda::array<dcomplex, 2>(r, r);
  for (int j = 0; j < r; ++j) {
    for (int k = 0; k < r; ++k) {
      lamb0jk(j, k) = 0;
      for (int l = 0; l < r; ++l) {
        if (l != j) {
          lamb0jk(j, k) += lambc(0, k, l) / (ej(j) - ej(l));
        }
      }
    }
  }

  // lamb0mj2(m,j) = sum_{k} lambda_kj / (i \omega_m - E_j - E_k)
  auto lamb0mj2 = nda::array<dcomplex, 2>(r, r);
  for (int m = 0; m < r; ++m) {
    for (int j = 0; j < r; ++j) {
      lamb0mj2(m, j) = 0;
      for (int k = 0; k < r; ++k) {
        lamb0mj2(m, j) += lambc(0, k, j) / (om_dlr(m) - ej(j) - ej(k));
      }
    }
  }

  // lamb0mj2sq(m,j) = sum_{k} lambda_kj / (i \omega_m - E_j - E_k)^2
  auto lamb0mj2sq = nda::array<dcomplex, 2>(r, r);
  for (int m = 0; m < r; ++m) {
    for (int j = 0; j < r; ++j) {
      lamb0mj2sq(m, j) = 0;
      for (int k = 0; k < r; ++k) {
        lamb0mj2sq(m, j) += lambc(0, k, j) / ((om_dlr(m) - ej(j) - ej(k)) *
                                              (om_dlr(m) - ej(j) - ej(k)));
      }
    }
  }

  // lamb1km(k,m) = sum_l lambda_kl / (i omega_m - E_l)
  auto lamb1km = nda::array<dcomplex, 2>(r, r);
  for (int k = 0; k < r; ++k) {
    for (int m = 0; m < r; ++m) {
      lamb1km(k, m) = 0;
      for (int l = 0; l < r; ++l) {
        lamb1km(k, m) += // lambc(1, k, l) / (om_dlr(m) - ej(l));
            lambc(1, k, l) * tanh(beta * ej(l) / 2) / (om_dlr(m) - ej(l));
      }
    }
  }

  // lamb2km(k,m) = sum_l lambda_kl / (i omega_m - E_l)
  auto lamb2km = nda::array<dcomplex, 2>(r, r);
  for (int k = 0; k < r; ++k) {
    for (int m = 0; m < r; ++m) {
      lamb2km(k, m) = 0;
      for (int l = 0; l < r; ++l) {
        lamb2km(k, m) += // lambc(2, k, l) / (om_dlr(m) - ej(l));
            lambc(2, k, l) * tanh(beta * ej(l) / 2) / (om_dlr(m) - ej(l));
      }
    }
  }

  // Compute polarization

  auto pol = nda::vector<dcomplex>(r);
  pol = 0;
  auto resvec = nda::vector<dcomplex>(r);
  std::complex<double> om = 0;

  // First term of DLR

  // z = E_j residues (double poles)

  // Simple pole contribution
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    auto tmp = hilbm(m, _, _) * lamb0jl;
    auto tmp2 = nda::array<dcomplex, 1>(r);
    for (int j = 0; j < r; ++j) {
      tmp2(j) = sum(tmp(j, _));
    }

    resvec =
        nf * nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), gc)) *
        (fc * tmp2 + nda::array_const_view<dcomplex, 1>(matvecmul(hilb, fc)) *
                         lamb0mj(m, _));
    pol(m) += sum(resvec);
  }

  // Double pole contribution
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    resvec =
        fc *
        (nfp *
             nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), gc)) *
             lamb0mj(m, _) +
         nf *
             nda::array_const_view<dcomplex, 1>(
                 matvecmul(hilbmsq(m, _, _), gc)) *
             lamb0mj(m, _) +
         nf *
             nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), gc)) *
             lamb0mjsq(m, _));
    pol(m) += sum(resvec);
  }

  // z = i omega_m - E_j residues (double poles)

  // Simple pole contribution
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    auto tmp = hilbm(m, _, _) * lamb0jk;
    auto tmp2 = nda::array<dcomplex, 1>(r);
    for (int j = 0; j < r; ++j) {
      tmp2(j) = sum(tmp(j, _));
    }

    resvec =
        nfm *
        nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), fc)) *
        (gc * tmp2 + nda::array_const_view<dcomplex, 1>(matvecmul(hilb, gc)) *
                         lamb0mj2(m, _));
    pol(m) -= sum(resvec);
  }

  // Double pole contribution
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    resvec =
        gc *
        (nfpm *
             nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), fc)) *
             lamb0mj2(m, _) -
         nfm *
             nda::array_const_view<dcomplex, 1>(
                 matvecmul(hilbmsq(m, _, _), fc)) *
             lamb0mj2(m, _) -
         nfm *
             nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), fc)) *
             lamb0mj2sq(m, _));
    pol(m) += sum(resvec);
  }

  // Second term of DLR

  // z = E_j residues
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    resvec = fc * nf *
             nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), gc)) *
             nda::array_const_view<dcomplex, 1>(
                 matvecmul(hilbm(m, _, _), lamb1km(_, m)));
    pol(m) += sum(resvec);
  }

  // z = i omega_m - E_j residues

  // Simple pole contribution
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    resvec = nfm *
             nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), fc)) *
             (gc * nda::array_const_view<dcomplex, 1>(
                       matvecmul(hilb, lamb1km(_, m))) +
              nda::array_const_view<dcomplex, 1>(matvecmul(hilb, gc)) *
                  nda::array_const_view<dcomplex, 1>(lamb1km(_, m)));
    pol(m) -= sum(resvec);
  }

  // Double pole contribution
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    resvec = (gc * nda::array_const_view<dcomplex, 1>(lamb1km(_, m))) *
             (nfpm * nda::array_const_view<dcomplex, 1>(
                         matvecmul(hilbm(m, _, _), fc)) -
              nfm * nda::array_const_view<dcomplex, 1>(
                        matvecmul(hilbmsq(m, _, _), fc)));
    pol(m) += sum(resvec);
  }

  // Third term of DLR

  // z = i omega_m E_j residues (simple poles)

  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    resvec = gc * nfm *
             nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), fc)) *
             nda::array_const_view<dcomplex, 1>(
                 matvecmul(hilbm(m, _, _), lamb2km(_, m)));
    pol(m) -= sum(resvec);
  }

  // z = E_j residues (double poles)

  // Simple pole contribution
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    resvec = nf *
             nda::array_const_view<dcomplex, 1>(matvecmul(hilbm(m, _, _), gc)) *
             (fc * nda::array_const_view<dcomplex, 1>(
                       matvecmul(hilb, lamb2km(_, m))) +
              nda::array_const_view<dcomplex, 1>(matvecmul(hilb, fc)) *
                  nda::array_const_view<dcomplex, 1>(lamb2km(_, m)));
    pol(m) += sum(resvec);
  }

  // Double pole contribution
  for (int m = 0; m < r; ++m) {
    om = om_dlr(m);
    if (dlr_if_bos(m) == 0) {
      continue;
    } // For now, skip zero frequency
    resvec = (fc * nda::array_const_view<dcomplex, 1>(lamb2km(_, m))) *
             (nfp * nda::array_const_view<dcomplex, 1>(
                        matvecmul(hilbm(m, _, _), gc)) +
              nf * nda::array_const_view<dcomplex, 1>(
                       matvecmul(hilbmsq(m, _, _), gc)));
    pol(m) += sum(resvec);
  }

  pol *= -1;

  // Compute polarization at i omega_n = 0

  // Evaluate summand at fermionic DLR imag freq nodes

  auto h = nda::vector<dcomplex>(r);
  std::complex<double> nu = 0;
  int channel = 1;
  for (int k = 0; k < r; ++k) {
    // h(k) = hubbg(u, nu1) * hubbg(u, -nu1) * hubbvert(u, beta, nu1, -nu1);
    h(k) = ifops_fer.coefs2eval(beta, fc, dlr_if_fer(k)) *
           ifops_fer.coefs2eval(beta, gc, -dlr_if_fer(k) - 1) *
           coefs2eval_if(beta, dlr_rf, lambc, lambc_sing, dlr_if_fer(k),
                         -dlr_if_fer(k) - 1, channel);
  }

  // Obtain DLR coefficients
  auto hc = ifops_fer.vals2coefs(beta, h);

  // Evaluate at tau = 0
  std::complex<double> pol0 = 0;
  for (int k = 0; k < r; ++k) {
    pol0 += hc(k) * k_it(0.0, ej(k), beta);
  }

  // Fill in polarization at i omega_n = 0
  int m0idx = 0;
  for (int m = 0; m < r; ++m) {
    if (dlr_if_bos(m) == 0) {
      m0idx = m;
      pol(m) = pol0;
    }
  }

  return pol;
}

// Compute contribution to polarization of constant part of vertex, assumed to
// be 1
nda::vector<dcomplex>
polarization_const(double beta, imtime_ops const &itops,
                   imfreq_ops const &ifops_bos,
                   nda::array_const_view<dcomplex, 1> fc,
                   nda::array_const_view<dcomplex, 1> gc) {

  auto fit = itops.coefs2vals(fc);
  auto git = itops.coefs2vals(gc);
  auto fgit = make_regular(fit * git);
  auto fgc = itops.vals2coefs(fgit);
  return beta * ifops_bos.coefs2vals(fgc);
}

} // namespace dlr2d
