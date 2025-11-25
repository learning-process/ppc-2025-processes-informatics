#include "kondakov_v_min_val_in_matrix_str/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"

namespace kondakov_v_min_val_in_matrix_str {

KondakovVMinValMatrixSEQ::KondakovVMinValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType(in.size());
}

bool KondakovVMinValMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  size_t cols = matrix[0].size();
  // NOLINTNEXTLINE(readability-use-anyofallof)
  for (const auto &row : matrix) {
    if (row.empty()) {
      return false;
    }
    if (row.size() != cols) {
      return false;
    }
  }
  return true;
}

bool KondakovVMinValMatrixSEQ::PreProcessingImpl() {
  return true;
}

bool KondakovVMinValMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &output = GetOutput();

  for (size_t i = 0; i < matrix.size(); ++i) {
    int min_val = std::numeric_limits<int>::max();
    for (int val : matrix[i]) {
      min_val = std::min(min_val, val);
    }
    output[i] = min_val;
  }

  return true;
}

bool KondakovVMinValMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kondakov_v_min_val_in_matrix_str
