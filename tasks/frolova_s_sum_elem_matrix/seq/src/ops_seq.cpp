#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wnull-dereference"
#endif

namespace frolova_s_sum_elem_matrix {

FrolovaSSumElemMatrixSEQ::FrolovaSSumElemMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FrolovaSSumElemMatrixSEQ::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  const std::size_t cols = matrix.front().size();
  if (cols == 0) {
    return false;
  }

  return std::ranges::all_of(matrix, [cols](const auto &row) { return row.size() == cols; });
}

bool FrolovaSSumElemMatrixSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool FrolovaSSumElemMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  int64_t sum = 0;

  for (const auto &row : matrix) {
    sum += std::accumulate(row.begin(), row.end(), static_cast<int64_t>(0));
  }

  GetOutput() = sum;
  return true;
}

bool FrolovaSSumElemMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_sum_elem_matrix
