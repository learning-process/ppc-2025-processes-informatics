#include "lukin_i_elem_vec_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "lukin_i_elem_vec_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace lukin_i_elem_vec_sum {

LukinIElemVecSumSEQ::LukinIElemVecSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LukinIElemVecSumSEQ::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();
}

bool LukinIElemVecSumSEQ::PreProcessingImpl() {
  vector_to_count = GetInput();

  elem_vec_sum = 0;

  return true;
}

bool LukinIElemVecSumSEQ::RunImpl() {
  for (const auto &elem : vector_to_count) {
    elem_vec_sum += elem;
  }
  return true;
}

bool LukinIElemVecSumSEQ::PostProcessingImpl() {
  GetOutput() = elem_vec_sum;
  return true;
}

}  // namespace lukin_i_elem_vec_sum
