#include "konstantinov_s_elem_vec_sign_change_count/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "konstantinov_s_elem_vec_sign_change_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

KonstantinovSElemVecSignChangeSEQ::KonstantinovSElemVecSignChangeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KonstantinovSElemVecSignChangeSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool KonstantinovSElemVecSignChangeSEQ::PreProcessingImpl() {
  return true;
}

bool KonstantinovSElemVecSignChangeSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  const auto invec = GetInput();
  
  for (int i=0; i<invec.size()-1;i++)
  {
    GetOutput() += (invec[i]>0) != (invec[i+1]>0); // + 1 если занки разные
  }
  return true;
}

bool KonstantinovSElemVecSignChangeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace konstantinov_s_elem_vec_sign_change_count
