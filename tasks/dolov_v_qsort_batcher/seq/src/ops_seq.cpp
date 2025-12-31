#include "dolov_v_qsort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>

#include "dolov_v_qsort_batcher/common/include/common.hpp"

namespace dolov_v_qsort_batcher {

DolovVQsortBatcherSEQ::DolovVQsortBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DolovVQsortBatcherSEQ::ValidationImpl() {
  return true;
}

bool DolovVQsortBatcherSEQ::PreProcessingImpl() {
  res_array_ = GetInput();
  return true;
}

bool DolovVQsortBatcherSEQ::RunImpl() {
  if (res_array_.empty()) {
    return true;
  }
  ApplyQuicksort(res_array_.data(), 0, static_cast<int>(res_array_.size()) - 1);
  return true;
}

bool DolovVQsortBatcherSEQ::PostProcessingImpl() {
  GetOutput() = std::move(res_array_);
  return true;
}

int DolovVQsortBatcherSEQ::GetHoarePartition(double *array, int low, int high) {
  double pivot = array[low + (high - low) / 2];
  int i = low - 1;
  int j = high + 1;

  while (true) {
    do {
      i++;
    } while (array[i] < pivot);
    do {
      j--;
    } while (array[j] > pivot);
    if (i >= j) {
      return j;
    }
    std::swap(array[i], array[j]);
  }
}

void DolovVQsortBatcherSEQ::ApplyQuicksort(double *array, int low, int high) {
  if (low < high) {
    int p = GetHoarePartition(array, low, high);

    ApplyQuicksort(array, low, p);
    ApplyQuicksort(array, p + 1, high);
  }
}

}  // namespace dolov_v_qsort_batcher
