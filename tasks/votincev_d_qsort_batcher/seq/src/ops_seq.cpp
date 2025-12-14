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
  std::vector<int> stack;

  stack.push_back(left);
  stack.push_back(right);

  while (!stack.empty()) {
    int h = stack.back();
    stack.pop_back();
    int l = stack.back();
    stack.pop_back();

    if (l >= h) {
      continue;
    }

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

    if (l < j) {
      stack.push_back(l);
      stack.push_back(j);
    }

    if (i < h) {
      stack.push_back(i);
      stack.push_back(h);
    }
  }
}

bool VotincevDQsortBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_qsort_batcher
