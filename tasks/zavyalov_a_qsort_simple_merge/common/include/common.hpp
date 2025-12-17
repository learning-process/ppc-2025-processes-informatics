#pragma once

#include <mpi.h>

#include <algorithm>  // TODO убрать
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>

#include "task/include/task.hpp"

namespace zavyalov_a_qsort_simple_merge {
using InType = std::vector<double>;   // сортируемый массив
using OutType = std::vector<double>;  // результат сортировки
using TestType = size_t;              // размер массива
using BaseTask = ppc::task::Task<InType, OutType>;

inline void MyQsort(double *mem, int left, int right) {
  if (left >= right || mem == nullptr) {
    return;
  }
  int l = left, r = right;
  int pivot_ind = left + rand() % (right - left + 1);
  double piv = mem[pivot_ind];
  while (l <= r) {
    while (mem[l] < piv) {
      l++;
    }

    while (mem[r] > piv) {
      r--;
    }

    if (l <= r) {
      std::swap(mem[l++], mem[r--]);
    }
  }
  if (left < r) {
    MyQsort(mem, left, r);
  }

  if (right > l) {
    MyQsort(mem, l, right);
  }
}

}  // namespace zavyalov_a_qsort_simple_merge
