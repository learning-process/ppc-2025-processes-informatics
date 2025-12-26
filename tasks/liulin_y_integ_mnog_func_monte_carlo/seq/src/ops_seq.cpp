#include "liulin_y_integ_mnog_func_monte_carlo/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "liulin_y_integ_mnog_func_monte_carlo/common/include/common.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

LiulinYIntegMnogFuncMonteCarloSEQ::LiulinYIntegMnogFuncMonteCarloSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  auto &matrix = std::get<0>(GetInput());
  auto &vect = std::get<1>(GetInput());

  const auto &input_matrix = std::get<0>(in);
  const auto &input_vect = std::get<1>(in);

  matrix.clear();
  vect.clear();

  if (!input_matrix.empty()) {
    matrix = input_matrix;
  }

  if (!input_vect.empty()) {
    vect = input_vect;
  }

  GetOutput().clear();
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &matrix = std::get<0>(input);
  const auto &vect = std::get<1>(input);

  if (matrix.empty() && vect.empty()) {
    return true;
  }

  return true;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  const auto &matrix = std::get<0>(input);

  if (matrix.empty() || matrix[0].empty()) {
    GetOutput().clear();
    return true;
  }

  const std::size_t rows = matrix.size();
  GetOutput().assign(rows, 0);
  return true;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::RunImpl() {
  const auto &input = GetInput();
  const auto &matrix = std::get<0>(input);
  const auto &vect = std::get<1>(input);
  auto &out = GetOutput();

  if (matrix.empty() || matrix[0].empty()) {
    return true;
  }

  const int rows = static_cast<int>(matrix.size());
  const int cols = static_cast<int>(matrix[0].size());

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out[i] += matrix[i][j] * vect[j];
    }
  }

  return true;
}

bool LiulinYIntegMnogFuncMonteCarloSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace liulin_y_integ_mnog_func_monte_carlo
