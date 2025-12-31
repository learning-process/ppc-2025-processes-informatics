#include "kiselev_i_gauss_method_horizontal_tape_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

KiselevITestTaskSEQ::KiselevITestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &buf = GetInput();
  InType tmp(in);
  buf.swap(tmp);
  GetOutput().clear();
}

bool KiselevITestTaskSEQ::ValidationImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());

  return !a.empty() && a.size() == b.size() && GetOutput().empty();
}

bool KiselevITestTaskSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool KiselevITestTaskSEQ::RunImpl() {
  auto mat = std::get<0>(GetInput());
  auto rhs = std::get<1>(GetInput());
  const std::size_t band = std::get<2>(GetInput());

  const std::size_t n = mat.size();
  GetOutput().assign(n, 0.0);

  constexpr double eps = 1e-12;

  for (std::size_t k = 0; k < n; ++k) {
    double diag = mat[k][k];
    if (std::fabs(diag) < eps) {
      return false;
    }

    std::size_t col_end = std::min(n, k + band + 1);

    for (std::size_t j = k; j < col_end; ++j) {
      mat[k][j] /= diag;
    }
    rhs[k] /= diag;

    std::size_t row_end = std::min(n, k + band + 1);
    for (std::size_t i = k + 1; i < row_end; ++i) {
      double factor = mat[i][k];
      if (factor == 0.0) {
        continue;
      }

      for (std::size_t j = k; j < col_end; ++j) {
        mat[i][j] -= factor * mat[k][j];
      }
      rhs[i] -= factor * rhs[k];
    }
  }

  for (std::size_t step = 0; step < n; ++step) {
    std::size_t i = n - 1 - step;
    double acc = rhs[i];

    std::size_t col_end = std::min(n, i + band + 1);
    for (std::size_t j = i + 1; j < col_end; ++j) {
      acc -= mat[i][j] * GetOutput()[j];
    }

    GetOutput()[i] = acc;
  }

  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
