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
  const auto &input = GetInput();
  if (input.empty() || input[0].empty()) {
    return false;
  }
  const size_t cols = input[0].size();
  return std::all_of(input.begin(), input.end(), [cols](const auto &row) { return row.size() == cols; });
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
