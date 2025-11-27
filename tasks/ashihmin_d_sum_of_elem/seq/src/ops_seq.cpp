#include "ashihmin_d_sum_of_elem/seq/include/ops_seq.hpp"

namespace ashihmin_d_sum_of_elem {

AshihminDElemVecsSumSEQ::AshihminDElemVecsSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AshihminDElemVecsSumSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool AshihminDElemVecsSumSEQ::PreProcessingImpl() {
  return true;
}

bool AshihminDElemVecsSumSEQ::RunImpl() {
  const auto &vec = GetInput();
  long long sum = 0;

  for (int v : vec) {
    sum += v;
  }

  GetOutput() = sum;
  return true;
}

bool AshihminDElemVecsSumSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ashihmin_d_sum_of_elem
