#include "morozova_s_matrix_max_value/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"

namespace morozova_s_matrix_max_value {

MorozovaSMatrixMaxValueSEQ::MorozovaSMatrixMaxValueSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0;
}

bool MorozovaSMatrixMaxValueSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }
  const size_t first_row_size = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != first_row_size) {
      return false;
    }
  }
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty() || matrix[0].empty()) {
    GetOutput() = std::numeric_limits<int>::min();
    return true;
  }
  const size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      GetOutput() = std::numeric_limits<int>::min();
      return true;
    }
  }
  int max_value = std::numeric_limits<int>::min();
  for (const auto &row : matrix) {
    for (int value : row) {
      max_value = std::max(max_value, value);
    }
  }
  GetOutput() = max_value;
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_matrix_max_value
