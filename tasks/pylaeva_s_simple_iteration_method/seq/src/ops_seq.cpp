#include "pylaeva_s_simple_iteration_method/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "pylaeva_s_simple_iteration_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace pylaeva_s_simple_iteration_method {

PylaevaSSimpleIterationMethodSEQ::PylaevaSSimpleIterationMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PylaevaSSimpleIterationMethodSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool PylaevaSSimpleIterationMethodSEQ::PreProcessingImpl() {
  
  return GetOutput() > 0;
}

bool PylaevaSSimpleIterationMethodSEQ::RunImpl() {
  
  return GetOutput() > 0;
}

bool PylaevaSSimpleIterationMethodSEQ::PostProcessingImpl() {
  
  return GetOutput() > 0;
}

}  // namespace pylaeva_s_simple_iteration_method
