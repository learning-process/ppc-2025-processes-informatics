#include "pylaeva_s_max_elem_matrix/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "pylaeva_s_max_elem_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace pylaeva_s_max_elem_matrix {

PylaevaSMaxElemMatrixSEQ::PylaevaSMaxElemMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PylaevaSMaxElemMatrixSEQ::ValidationImpl() {
  return (static_cast<size_t>(std::get<0>(GetInput()))==std::get<1>(GetInput()).size()) && (static_cast<size_t>(std::get<0>(GetInput()))>0) && (GetOutput() == 0);
}

bool PylaevaSMaxElemMatrixSEQ::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::min();
  return true;
}

bool PylaevaSMaxElemMatrixSEQ::RunImpl() {
  const auto& matrix_data = std::get<1>(GetInput());
  const auto& matrix_size = std::get<0>(GetInput());
  
  if (matrix_data.empty() || matrix_size == 0 || matrix_data.size()!=matrix_size) {
    return false;
  }

  int max_element = matrix_data[0];  
  
  for (size_t i = 1; i < std::get<0>(GetInput()); ++i) {
    if (std::get<1>(GetInput())[i] > max_element) {
      max_element = std::get<1>(GetInput())[i];
    }
  }
  
  GetOutput() = max_element;
  return true;
}

bool PylaevaSMaxElemMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace pylaeva_s_max_elem_matrix
