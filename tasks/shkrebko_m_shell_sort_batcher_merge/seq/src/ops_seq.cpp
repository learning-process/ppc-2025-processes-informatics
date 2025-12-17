#include "shkrebko_m_shell_sort_batcher_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "shkrebko_m_shell_sort_batcher_merge/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

ShkrebkoMShellSortBatcherMergeSEQ::ShkrebkoMShellSortBatcherMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShkrebkoMShellSortBatcherMergeSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool ShkrebkoMShellSortBatcherMergeSEQ::PreProcessingImpl() {
  return true;
}

void ShkrebkoMShellSortBatcherMergeSEQ::ShellSort(std::vector<int> &arr) {
  size_t n = arr.size();
  if (n <= 1) return;

  std::vector<int> gaps;
  int k = 0;
  int current_gap;
  do {
    if (k % 2 == 0) {
      current_gap = static_cast<int>(9 * std::pow(2, k) - 9 * std::pow(2, k / 2) + 1);
    } else {
      current_gap = static_cast<int>(8 * std::pow(2, k) - 6 * std::pow(2, (k + 1) / 2) + 1);
    }
    if (current_gap > 0 && static_cast<size_t>(current_gap) < n) {
      gaps.push_back(current_gap);
    }
    k++;
  } while (current_gap > 0 && static_cast<size_t>(current_gap) < n);

  std::sort(gaps.rbegin(), gaps.rend());
  
  for (int gap_val : gaps) {
    for (size_t i = gap_val; i < n; ++i) {
      int temp = arr[i];
      size_t j = i;
      while (j >= static_cast<size_t>(gap_val) && arr[j - gap_val] > temp) {
        arr[j] = arr[j - gap_val];
        j -= gap_val;
      }
      arr[j] = temp;
    }
  }
}

bool ShkrebkoMShellSortBatcherMergeSEQ::RunImpl() {
  GetOutput() = GetInput();
  ShellSort(GetOutput());
  return !GetOutput().empty();
}

bool ShkrebkoMShellSortBatcherMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkrebko_m_shell_sort_batcher_merge