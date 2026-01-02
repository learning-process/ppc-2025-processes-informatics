#include "shvetsova_k_rad_sort_batch_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace shvetsova_k_rad_sort_batch_merge {

ShvetsovaKRadSortBatchMergeSEQ::ShvetsovaKRadSortBatchMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ShvetsovaKRadSortBatchMergeSEQ::ValidationImpl() {
  data_ = GetInput();
  return true;
}

bool ShvetsovaKRadSortBatchMergeSEQ::PreProcessingImpl() {
  return true;
}

bool ShvetsovaKRadSortBatchMergeSEQ::RunImpl() {
  if (data_.empty()) {
    GetOutput().clear();
    return true;
  }

  // Поразрядная сортировка
  RadixSort(data_);

  // Чётно-нечётное слияние Бэтчера
  BatcherOddEvenMergeSort(data_, 0, static_cast<int>(data_.size()));

  GetOutput().assign(data_.begin(), data_.end());
  return true;
}

bool ShvetsovaKRadSortBatchMergeSEQ::PostProcessingImpl() {
  return true;
}

/* ===================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===================== */

// LSD Radix Sort для целых чисел в double
void ShvetsovaKRadSortBatchMergeSEQ::RadixSort(std::vector<double> &vec) {
  if (vec.empty()) {
    return;
  }
  const int base = 10;
  int max_val = 0;

  for (double x : vec) {
    max_val = std::max(max_val, static_cast<int>(std::abs(x)));
  }

  for (int exp = 1; max_val / exp > 0; exp *= base) {
    std::vector<double> output(vec.size());
    std::array<int, base> count{};

    for (double x : vec) {
      int digit = (static_cast<int>(std::abs(x)) / exp) % base;
      count.at(digit)++;
    }

    for (int i = 1; i < base; i++) {
      count.at(i) += count.at(i - 1);
    }

    for (int i = static_cast<int>(vec.size()) - 1; i >= 0; i--) {
      int digit = (static_cast<int>(std::abs(vec[i])) / exp) % base;
      output.at(--count.at(digit)) = vec.at(i);
    }

    vec = std::move(output);
  }
}

// Сортировка Бэтчера (odd-even merge sort)
void ShvetsovaKRadSortBatchMergeSEQ::BatcherOddEvenMergeSort(std::vector<double> &vec, int left, int right) {
  if (right - left <= 1) {
    return;
  }

  int mid = (left + right) / 2;

  BatcherOddEvenMergeSort(vec, left, mid);
  BatcherOddEvenMergeSort(vec, mid, right);

  OddEvenMerge(vec, left, right, 1);
}

// Чётно-нечётное слияние
void ShvetsovaKRadSortBatchMergeSEQ::OddEvenMerge(std::vector<double> &vec, int left, int right, int step) {
  int dist = step * 2;

  if (dist < right - left) {
    OddEvenMerge(vec, left, right, dist);
    OddEvenMerge(vec, left + step, right, dist);

    for (int i = left + step; i + step < right; i += dist) {
      if (vec.at(i) > vec.at(i + step)) {
        std::swap(vec.at(i), vec.at(i + step));
      }
    }
  } else {
    if (left + step < right && vec.at(left) > vec.at(left + step)) {
      std::swap(vec.at(left), vec.at(left + step));
    }
  }
}

}  // namespace shvetsova_k_rad_sort_batch_merge
