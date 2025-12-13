#include "nikitina_v_quick_sort_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "nikitina_v_quick_sort_merge/common/include/common.hpp"

namespace nikitina_v_quick_sort_merge {

namespace {
void QuickSortImpl(std::vector<int> &vec, int left, int right) {
  int i = left;
  int j = right;
  int pivot = vec[(left + right) / 2];
  while (i <= j) {
    while (vec[i] < pivot) {
      i++;
    }
    while (vec[j] > pivot) {
      j--;
    }
    if (i <= j) {
      std::swap(vec[i], vec[j]);
      i++;
      j--;
    }
  }
  if (left < j) {
    QuickSortImpl(vec, left, j);
  }
  if (i < right) {
    QuickSortImpl(vec, i, right);
  }
}
}  // namespace

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TestTaskSEQ::ValidationImpl() {
  return true;
}

bool TestTaskSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool TestTaskSEQ::RunImpl() {
  if (GetOutput().empty()) {
    return true;
  }
  QuickSortImpl(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);
  return true;
}

bool TestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_quick_sort_merge
