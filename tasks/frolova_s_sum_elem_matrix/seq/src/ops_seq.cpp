#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"

#include <numeric>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

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

  // все строки должны быть не пустыми и одинаковой длины
  const std::size_t cols = matrix.front().size();
  if (cols == 0) {
    return false;
  }

  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool FrolovaSSumElemMatrixSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool FrolovaSSumElemMatrixSEQ::RunImpl() {
  const auto &matrix = GetInput();
  long long sum = 0;

  for (const auto &row : matrix) {
    sum += std::accumulate(row.begin(), row.end(), 0LL);
  }

  GetOutput() = sum;
  return true;
}

bool FrolovaSSumElemMatrixSEQ::PostProcessingImpl() {
  // Ничего дополнительно делать не нужно
  return true;
}

}  // namespace frolova_s_sum_elem_matrix

