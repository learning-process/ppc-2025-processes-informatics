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
  return true;
}

bool LukinIElemVecSumSEQ::RunImpl() {
  const auto &vec = GetInput();
  int sum = 0;
  for (const auto &elem : vec) {
    sum += elem;
  }
  GetOutput() = sum;
  return true;
}

bool LukinIElemVecSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_elem_vec_sum
