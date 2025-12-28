#pragma once

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace maslova_u_fast_sort_simple {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline void QuickSort(int *data, int left, int right) {
  if (data == nullptr || left >= right) {
    return;
  }

  std::vector<std::pair<int, int>> stack;
  stack.reserve(64);
  stack.emplace_back(left, right);

  while (!stack.empty()) {
    auto [low, high] = stack.back();
    stack.pop_back();

    int i = low;
    int j = high;
    int pivot = data[low + (high - low) / 2];

    while (i <= j) {
      while (data[i] < pivot) {
        i++;
      }
      while (data[j] > pivot) {
        j--;
      }
      if (i <= j) {
        std::swap(data[i], data[j]);
        i++;
        j--;
      }
    }

    if (high - i > j - low) {
      if (low < j) {
        stack.emplace_back(low, j);
      }
      if (i < high) {
        stack.emplace_back(i, high);
      }
    } else {
      if (i < high) {
        stack.emplace_back(i, high);
      }
      if (low < j) {
        stack.emplace_back(low, j);
      }
    }
  }
}

}  // namespace maslova_u_fast_sort_simple
