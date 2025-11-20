#include "ashihmin_d_sum_of_elem/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "ashihmin_d_sum_of_elem/common/include/common.hpp"
#include "util/include/util.hpp"

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
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

// bool AshihminDElemVecsSumSEQ::RunImpl() { //решение тут1
//   if (GetInput() == 0) {
//     return false;
//   }

//   for (InType i = 0; i < GetInput(); i++) {
//     for (InType j = 0; j < GetInput(); j++) {
//       for (InType k = 0; k < GetInput(); k++) {
//         std::vector<InType> tmp(i + j + k, 1);
//         GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
//         GetOutput() -= i + j + k;
//       }
//     }
//   }

//   const int num_threads = ppc::util::GetNumThreads();
//   GetOutput() *= num_threads;

//   int counter = 0;
//   for (int i = 0; i < num_threads; i++) {
//     counter++;
//   }

//   if (counter != 0) {
//     GetOutput() /= counter;
//   }
//   return GetOutput() > 0;
// }

// bool AshihminDElemVecsSumSEQ::PostProcessingImpl() {
//   GetOutput() -= GetInput();
//   return GetOutput() > 0;
// }
bool AshihminDElemVecsSumSEQ::RunImpl() {
  int n = GetInput();
  if (n <= 0) {
    return false;
  }

  int sum = 0;
  for (int i = 0; i < n; i++) {
    sum += 1;  // создаем "виртуальный вектор" единиц
  }

  GetOutput() = sum;
  return true;
}

bool AshihminDElemVecsSumSEQ::PostProcessingImpl() {
  return true;  // output уже равен input
}

}  // namespace ashihmin_d_sum_of_elem
