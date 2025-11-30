#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <ranges>  // NOLINT(misc-include-cleaner)
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

  std::size_t row_idx = 0;
  for (const auto &row : matrix) {
    int tmp_max = row[0];
    for (int val : row) {
      tmp_max = std::max(val, tmp_max);
    }
    out_vector[row_idx++] = tmp_max;
  }
  return true;
}

bool KiselevITestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_max_value_in_strings
