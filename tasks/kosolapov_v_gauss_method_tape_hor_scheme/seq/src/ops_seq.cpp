#include "kosolapov_v_gauss_method_tape_hor_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

KosolapovVGaussMethodTapeHorSchemeSEQ::KosolapovVGaussMethodTapeHorSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.matrix.size() <= 0) {
    return false;
  }
  int n = static_cast<int>(input.matrix.size());
  if (input.matrix.size() != static_cast<size_t>(n)) {
    return false;
  }
  for (size_t i = 0; i < input.matrix.size(); ++i) {
    if (input.matrix[i].size() != static_cast<size_t>(n)) {
      return false;
    }
  }
  if (input.r_side.size() != static_cast<size_t>(n)) {
    return false;
  }
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  int n = static_cast<int>(input.matrix.size());
  GetOutput() = std::vector<double>(n, 0.0);
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();
  int n = static_cast<int>(input.matrix.size());

  std::vector<std::vector<double>> A = input.matrix;
  std::vector<double> b = input.r_side;
  std::vector<int> col_order(n);
  std::iota(col_order.begin(), col_order.end(), 0);
  for (int i = 0; i < n; ++i) {
    int leading_col = i;
    double max_elem = std::abs(A[i][i]);
    for (int j = i + 1; j < n; ++j) {
      if (std::abs(A[i][j]) > max_elem) {
        max_elem = std::abs(A[i][j]);
        leading_col = j;
      }
    }
    if (leading_col != i) {
      for (int k = 0; k < n; ++k) {
        std::swap(A[k][i], A[k][leading_col]);
      }
      std::swap(col_order[i], col_order[leading_col]);
    }
    double cur_el = A[i][i];
    for (int j = i; j < n; ++j) {
      A[i][j] /= cur_el;
    }
    b[i] /= cur_el;
    for (int k = i + 1; k < n; ++k) {
      double ratio = A[k][i];
      for (int j = i; j < n; ++j) {
        A[k][j] -= ratio * A[i][j];
      }
      b[k] -= ratio * b[i];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    output[col_order[i]] = b[i];
    for (int j = i + 1; j < n; ++j) {
      output[col_order[i]] -= A[i][j] * output[col_order[j]];
    }
  }
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kosolapov_v_gauss_method_tape_hor_scheme
