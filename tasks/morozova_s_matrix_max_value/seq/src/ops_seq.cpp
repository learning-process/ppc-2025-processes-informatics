#include "morozova_s_matrix_max_value/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "morozova_s_matrix_max_value/common/include/common.hpp"

namespace morozova_s_matrix_max_value {

MorozovaSMatrixMaxValueSEQ::MorozovaSMatrixMaxValueSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MorozovaSMatrixMaxValueSEQ::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty() && (GetOutput() == 0);
}

bool MorozovaSMatrixMaxValueSEQ::PreProcessingImpl() {
  GetOutput() = GetInput()[0][0];
  return true;
}

bool MorozovaSMatrixMaxValueSEQ::RunImpl() {
  auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
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
  return GetOutput() != std::numeric_limits<int>::min();
}

}  // namespace morozova_s_matrix_max_value
