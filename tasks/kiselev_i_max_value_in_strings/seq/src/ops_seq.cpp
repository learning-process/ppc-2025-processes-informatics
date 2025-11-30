#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"

#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"

namespace kiselev_i_max_value_in_strings {

KiselevITestTaskSEQ::KiselevITestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
  GetOutput().clear();
}

bool KiselevITestTaskSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  return std::ranges::all_of(matrix, [](const auto &rw) { return !rw.empty(); });
}

bool KiselevITestTaskSEQ::PreProcessingImpl() {
  const auto &matrix = GetInput();
  GetOutput().resize(matrix.size());
  return true;
}

bool KiselevITestTaskSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &out_vector = GetOutput();

  for (std::size_t row_idx = 0; row_idx < matrix.size(); ++row_idx) {
    int tmp_max = matrix[row_idx][0];
    for (std::size_t col_idx = 0; col_idx < matrix[row_idx].size(); ++col_idx) {
      tmp_max = std::max(matrix[row_idx][col_idx], tmp_max);
    }
    out_vector[row_idx] = tmp_max;
  }
  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_max_value_in_strings
