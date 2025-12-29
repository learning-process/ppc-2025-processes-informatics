#include "goriacheva_k_violation_order_elem_vec/seq/include/ops_seq.hpp"

//#include <numeric>
#include <vector>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace goriacheva_k_violation_order_elem_vec {

GoriachevaKViolationOrderElemVecSEQ::GoriachevaKViolationOrderElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GoriachevaKViolationOrderElemVecSEQ::ValidationImpl() {
  return true;
}

bool GoriachevaKViolationOrderElemVecSEQ::PreProcessingImpl() {
  input_vec = GetInput();
  result = 0;
  return true;
}

bool GoriachevaKViolationOrderElemVecSEQ::RunImpl() {
  input_vec = GetInput();
  result = 0;

  if (input_vec.size() <= 1) { 
    result = 0; 
  }

  for(size_t i = 0; i + 1 < input_vec.size(); ++i){
    if (input_vec[i] > input_vec[i + 1]){
      ++result;
    }
  }
  //GetOutput() = result;
  return true;
}

bool GoriachevaKViolationOrderElemVecSEQ::PostProcessingImpl() {
  GetOutput() = result;
  return true;
}

}  // namespace goriacheva_k_violation_order_elem_vec
