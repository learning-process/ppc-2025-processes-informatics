#include "ashihmin_d_sum_of_elem/seq/include/ops_seq.hpp"

#include "ashihmin_d_sum_of_elem/common/include/common.hpp"

namespace ashihmin_d_sum_of_elem {

AshihminDElemVecsSumSEQ::AshihminDElemVecsSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AshihminDElemVecsSumSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool AshihminDElemVecsSumSEQ::PreProcessingImpl() {
  return true;
}

bool AshihminDElemVecsSumSEQ::RunImpl() {
  const int n = GetInput();
  if (n <= 0) {
    return false;
  }

  int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += 1;
  }

  GetOutput() = sum;
  return true;
}

bool AshihminDElemVecsSumSEQ::PostProcessingImpl() {
  return GetOutput() == GetInput();
}

}  // namespace ashihmin_d_sum_of_elem
