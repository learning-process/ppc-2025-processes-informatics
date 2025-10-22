#include "guseva_a_matrix_sums/seq/include/ops_seq.hpp"

#include <ranges>

namespace guseva_a_matrix_sums {

GusevaAMatrixSumsSEQ::GusevaAMatrixSumsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool GusevaAMatrixSumsSEQ::ValidationImpl() {
  return (static_cast<uint64_t>(std::get<0>(GetInput())) * std::get<1>(GetInput()) == std::get<2>(GetInput()).size()) &&
         (GetOutput().empty());
}

bool GusevaAMatrixSumsSEQ::PreProcessingImpl() {
  GetOutput().resize(std::get<1>(GetInput()), 0.0);
  return true;
}

bool GusevaAMatrixSumsSEQ::RunImpl() {
  auto &matrix = std::get<2>(GetInput());
  auto &columns = std::get<1>(GetInput());
#if defined(__APPLE__) && defined(__clang__)
  auto &rows = std::get<0>(GetInput());
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < columns; j++) {
      GetOutput()[j] += matrix[(i * columns) + j];
    }
  }
#else
  for (const auto &[column, x] : GetOutput() | std::views::enumerate) {
    const auto &arr = matrix | std::ranges::views::drop(column) | std::views::stride(columns);
    x = std::reduce(arr.begin(), arr.end(), 0.0);
  }
#endif
  return true;
}

bool GusevaAMatrixSumsSEQ::PostProcessingImpl() {
  return GetOutput().size() > 0;
}

}  // namespace guseva_a_matrix_sums
