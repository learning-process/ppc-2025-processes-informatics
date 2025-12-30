#include "dolov_v_qsort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "dolov_v_qsort_batcher/common/include/common.hpp"

namespace dolov_v_qsort_batcher {

DolovVQsortBatcherSEQ::DolovVQsortBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DolovVQsortBatcherSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool DolovVQsortBatcherSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool DolovVQsortBatcherSEQ::RunImpl() {
  auto &data = GetOutput();
  applyQuicksort(data.data(), 0, static_cast<int>(data.size()) - 1);
  return true;
}

int DolovVQsortBatcherSEQ::getHoarePartition(double *array, int low, int high) {
  double pivot = array[low + (high - low) / 2];
  int i = low - 1;
  int j = high + 1;

  while (true) {
    while (array[++i] < pivot);
    while (array[--j] > pivot);

    if (i >= j) {
      return j;
    }

    std::swap(array[i], array[j]);
  }
}

void DolovVQsortBatcherSEQ::applyQuicksort(double *array, int low, int high) {
  if (low < high) {
    int p = getHoarePartition(array, low, high);
    applyQuicksort(array, low, p);
    applyQuicksort(array, p + 1, high);
  }
}

bool DolovVQsortBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dolov_v_qsort_batcher
