#include "pylaeva_s_max_elem_matrix/seq/include/ops_seq.hpp"

#include <algorithm>  // для std::max
#include <cstddef>    // для size_t
#include <limits>     // для std::numeric_limits
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"

namespace pylaeva_s_max_elem_matrix {

PylaevaSMaxElemMatrixSEQ::PylaevaSMaxElemMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PylaevaSMaxElemMatrixSEQ::ValidationImpl() {
  return (static_cast<size_t>(std::get<0>(GetInput())) == std::get<1>(GetInput()).size()) &&
         (static_cast<size_t>(std::get<0>(GetInput())) > 0) && (GetOutput() == 0);
}

bool PylaevaSMaxElemMatrixSEQ::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool PylaevaSMaxElemMatrixSEQ::RunImpl() {
  const auto &matrix_size = static_cast<size_t>(std::get<0>(GetInput()));
  const auto &matrix_data = std::get<1>(GetInput());

  if (matrix_data.empty() || matrix_size == 0 || matrix_data.size() != matrix_size) {
    return false;
  }

  int max_element = matrix_data[0];

  for (size_t i = 1; i < matrix_size; ++i) {
    max_element = std::max(matrix_data[i], max_element);
  }

  GetOutput() = max_element;
  return true;
}

bool PylaevaSMaxElemMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pylaeva_s_max_elem_matrix
