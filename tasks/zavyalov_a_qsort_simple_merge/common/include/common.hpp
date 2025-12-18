#pragma once

#include <algorithm>
#include <cstring>
#include <ctime>
#include <random>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace zavyalov_a_qsort_simple_merge {
using InType = std::vector<double>;   // сортируемый массив
using OutType = std::vector<double>;  // результат сортировки
using TestType = size_t;              // размер массива
using BaseTask = ppc::task::Task<InType, OutType>;

inline void InnerCycle(double* mem, int& l, int& r, double piv) {
    while (mem[l] < piv) {
      ++l;
    }
    while (mem[r] > piv) {
      --r;
    }
    if (l <= r) {
      std::swap(mem[l++], mem[r--]);
    }
}

inline void MyQsort(double *mem, int left, int right) {
  if (mem == nullptr || left >= right) {
    return;
  }

  static std::mt19937 gen(time({}));

  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(left, right);

  while (!stack.empty()) {
    auto [cur_l, cur_r] = stack.back();
    stack.pop_back();

    int l = cur_l;
    int r = cur_r;

    std::uniform_int_distribution<int> distrib(l, r);
    int pivot_ind = distrib(gen);
    double piv = mem[pivot_ind];

    while (l <= r) {
      InnerCycle(mem, l, r, piv);
    }

    if (cur_l < r) {
      stack.emplace_back(cur_l, r);
    }
    if (l < cur_r) {
      stack.emplace_back(l, cur_r);
    }
  }
}

}  // namespace zavyalov_a_qsort_simple_merge
