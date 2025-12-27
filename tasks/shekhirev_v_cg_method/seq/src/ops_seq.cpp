#include "../include/ops_seq.hpp"

#include <cmath>
#include <numeric>
#include <vector>

namespace shekhirev_v_cg_method_seq {

ConjugateGradientSeq::ConjugateGradientSeq(const shekhirev_v_cg_method::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ConjugateGradientSeq::ValidationImpl() {
  const int n = GetInput().n;
  return n > 0 && GetInput().A.size() == static_cast<size_t>(n * n) && GetInput().b.size() == static_cast<size_t>(n);
}

bool ConjugateGradientSeq::PreProcessingImpl() {
  return true;
}

bool ConjugateGradientSeq::RunImpl() {
  const auto &A = GetInput().A;
  const auto &b = GetInput().b;
  const int n = GetInput().n;

  std::vector<double> x(n, 0.0);
  std::vector<double> r = b;
  std::vector<double> p = r;

  double rs_old = 0.0;
  for (const double val : r) {
    rs_old += val * val;
  }

  const int max_iter = n * 2;
  const double epsilon = 1e-10;

  for (int k = 0; k < max_iter; ++k) {
    std::vector<double> Ap(n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        Ap[i] += A[i * n + j] * p[j];
      }
    }

    double pAp = 0.0;
    for (int i = 0; i < n; ++i) {
      pAp += p[i] * Ap[i];
    }

    const double alpha = rs_old / pAp;

    for (int i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double rs_new = 0.0;
    for (const double val : r) {
      rs_new += val * val;
    }

    if (std::sqrt(rs_new) < epsilon) {
      break;
    }

    const double beta = rs_new / rs_old;

    for (int i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }

  GetOutput() = x;
  return true;
}

bool ConjugateGradientSeq::PostProcessingImpl() {
  return true;
}

}  // namespace shekhirev_v_cg_method_seq
