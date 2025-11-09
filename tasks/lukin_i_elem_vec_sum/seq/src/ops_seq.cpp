#include "lukin_i_elem_vec_sum/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "lukin_i_elem_vec_sum/common/include/common.hpp"

namespace lukin_i_elem_vec_sum {

LukinIElemVecSumSEQ::LukinIElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LukinIElemVecSumSEQ::ValidationImpl() {
  return true;
}

bool LukinIElemVecSumSEQ::PreProcessingImpl() {
  vec_size = static_cast<size_t>(GetInput().size());
  return true;
}

bool LukinIElemVecSumSEQ::RunImpl() {
  if (vec_size == 0) {
    GetOutput() = 0;
    return true;
  }

  OutType sum = std::accumulate(GetInput().begin(), GetInput().end(), 0);

  GetOutput() = sum;
  return true;
}

bool LukinIElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_elem_vec_sum
