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
  return GetOutput()<0;
}

bool PylaevaSMaxElemMatrixSEQ::RunImpl() {
  int max_element = std::get<1>(GetInput())[0];  
  
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
