#include "votincev_d_qsort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>  // std::swap
#include <vector>

namespace votincev_d_qsort_batcher {

// --------------------
// ручная быстрая сортировка
// --------------------
void VotincevDQsortBatcherSEQ::QuickSort(double *arr, int left, int right) {
  int i = left;
  int j = right;
  double pivot = arr[(left + right) / 2];

  while (i <= j) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      j--;
    }
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }

  if (left < j) {
    QuickSort(arr, left, j);
  }
  if (i < right) {
    QuickSort(arr, i, right);
  }
}

VotincevDQsortBatcherSEQ::VotincevDQsortBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

// входные данные — просто std::vector<double>, проверяем, что не пустой
bool VotincevDQsortBatcherSEQ::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();
}

bool VotincevDQsortBatcherSEQ::PreProcessingImpl() {
  return true;
}

bool VotincevDQsortBatcherSEQ::RunImpl() {
  std::vector<double> data = GetInput();

  if (!data.empty()) {
    QuickSort(data.data(), 0, static_cast<int>(data.size()) - 1);
  }

  GetOutput() = data;
  return true;
}

bool VotincevDQsortBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
