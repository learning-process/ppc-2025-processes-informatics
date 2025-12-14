#include "votincev_d_qsort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

namespace votincev_d_qsort_batcher {

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

// итеративная qsort
void VotincevDQsortBatcherSEQ::QuickSort(double *arr, int left, int right) {
  std::vector<int> stack(right - left + 1);
  int top = -1;
  stack[++top] = left;
  stack[++top] = right;
  while (top >= 0) {
    int h = stack[top--];
    int l = stack[top--];
    int i = l;
    int j = h;
    double pivot = arr[(l + h) / 2];
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
    int p = i;
    if (l < j) {
      stack[++top] = l;
      stack[++top] = j;
    }
    if (p < h) {
      stack[++top] = p;
      stack[++top] = h;
    }
  }
}

bool VotincevDQsortBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
