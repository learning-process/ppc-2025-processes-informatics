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
    return false;
  }
  for (size_t i = 0; i < matrix.size(); ++i) {
    if (matrix[i].empty()) {
      return false;
    }
    if (i > 0 && matrix[i].size() != matrix[0].size()) {
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
