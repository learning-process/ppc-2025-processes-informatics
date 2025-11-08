#include "gutyansky_a_matrix_column_sum/seq/include/ops_seq.hpp"

#include <iostream>
#include <numeric>
#include <vector>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_matrix_column_sum {

GutyanskyAMatrixColumnSumSEQ::GutyanskyAMatrixColumnSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = {};
}

bool GutyanskyAMatrixColumnSumSEQ::ValidationImpl() {
  return GetInput().rows > 0 && GetInput().cols > 0 && GetInput().data.size() == GetInput().rows * GetInput().cols;
}

bool GutyanskyAMatrixColumnSumSEQ::PreProcessingImpl() {
  GetOutput().size = GetInput().cols;
  GetOutput().data.resize(GetInput().cols);

  return GetOutput().data.size() == GetInput().cols;
}

bool GutyanskyAMatrixColumnSumSEQ::RunImpl() {
  if (GetInput().rows == 0 || GetInput().cols == 0) {
    return false;
  }

  std::fill(GetOutput().data.begin(), GetOutput().data.end(), (int64_t)0);

  for (size_t i = 0; i < GetInput().rows * GetInput().cols; i++) {
    GetOutput().data[i % GetInput().cols] += GetInput().data[i];
  }

  return true;
}

bool GutyanskyAMatrixColumnSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_matrix_column_sum
