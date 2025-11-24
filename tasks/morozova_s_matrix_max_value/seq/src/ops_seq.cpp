#include "morozova_s_matrix_max_value/seq/include/ops_seq.hpp"

#include <algorithm>
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
  const auto &input = GetInput();
  if (input.empty() || input[0].empty()) {
    return false;
  }
  int cols = input[0].size();
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
  }
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::PreProcessingImpl() {
  GetOutput() = GetInput()[0][0];
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::RunImpl() {
  auto &matrix = GetInput();
  int max_value = std::numeric_limits<int>::min();
  for (const auto &row : matrix) {
    for (int value : row) {
      if (value > max_value) {
        max_value = value;
      }
    }
  }
  GetOutput() = max_value;
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_matrix_max_value
