#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace nikitina_v_quick_sort_merge {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline void QuickSortImpl(std::vector<int> &vec, int left, int right) {
  if (left >= right) {
    return;
  }

  std::vector<std::pair<int, int>> stack;
  stack.reserve(static_cast<size_t>(right - left + 1));
  stack.emplace_back(left, right);

  while (!stack.empty()) {
    auto [l, r] = stack.back();
    stack.pop_back();

    int i = l;
    int j = r;
    int pivot = vec[(l + r) / 2];

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

    if (l < j) {
      stack.emplace_back(l, j);
    }
    if (i < r) {
      stack.emplace_back(i, r);
    }
  }
}

}  // namespace nikitina_v_quick_sort_merge
