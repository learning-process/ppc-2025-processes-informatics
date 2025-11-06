#include "kurpiakov_a_elem_vec_sum//seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "kurpiakov_a_elem_vec_sum//common/include/common.hpp"
#include "util/include/util.hpp"

namespace kurpiakov_a_elem_vec_sum {

KurpiakovAElemVecSumSEQ::NesterovATestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool KurpiakovAElemVecSumSEQ::ValidationImpl() {
  bool if_dividable = std::get<1>(GetInput()).size() % std::get<0>(GetInput()) == 0;
  bool res = (GetOutput() == 0.0) && if_dividable;
  return res;
}

bool KurpiakovAElemVecSumSEQ::PreProcessingImpl() {
  GetOutput() == 0.0;
  return true;
}

bool KurpiakovAElemVecSumSEQ::RunImpl() {
  vec = std::get<1>(GetInput());
  GetOutput() = std::accumulate(vec.begin(), vec.end(), 0.0)
  return true;
}

bool KurpiakovAElemVecSumSEQ::PostProcessingImpl() {
  return true;
}


}  // namespace kurpiakov_a_elem_vec_sum
