#include "kiselev_i_gauss_method_horizontal_tape_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
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
  const auto &a_v = std::get<0>(GetInput());
  const auto &b_v = std::get<1>(GetInput());

  return !a_v.empty() && a_v.size() == b_v.size() && GetOutput().empty();
}

bool KiselevITestTaskSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool KiselevITestTaskSEQ::RunImpl() {
  auto mat = std::get<0>(GetInput());
  auto rhs = std::get<1>(GetInput());
  const std::size_t band = std::get<2>(GetInput());

  const std::size_t num = mat.size();
  GetOutput().assign(num, 0.0);

  constexpr double Eps = 1e-12;

  for (std::size_t k_index = 0; k_index < num; ++k_index) {
    double diag = mat[k_index][k_index];
    if (std::fabs(diag) < Eps) {
      return false;
    }

    std::size_t col_end = std::min(num, k_index + band + 1);

    for (std::size_t j_index = k_index; j_index < col_end; ++j_index) {
      mat[k_index][j_index] /= diag;
    }
    rhs[k_index] /= diag;

    std::size_t row_end = std::min(num, k_index + band + 1);
    for (std::size_t index = k_index + 1; index < row_end; ++index) {
      double factor = mat[index][k_index];
      if (factor == 0.0) {
        continue;
      }

      for (std::size_t j_index = k_index; j_index < col_end; ++j_index) {
        mat[index][j_index] -= factor * mat[k_index][j_index];
      }
      rhs[index] -= factor * rhs[k_index];
    }
  }

  for (std::size_t step = 0; step < num; ++step) {
    std::size_t index = num - 1 - step;
    double acc = rhs[index];

    std::size_t col_end = std::min(num, index + band + 1);
    for (std::size_t j_index = index + 1; j_index < col_end; ++j_index) {
      acc -= mat[index][j_index] * GetOutput()[j_index];
    }

    GetOutput()[index] = acc;
  }

  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
