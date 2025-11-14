#include "sizov_d_bubble_sort/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"

namespace sizov_d_bubble_sort {

SizovDBubbleSortSEQ::SizovDBubbleSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return !input.empty();
}

bool SizovDBubbleSortSEQ::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool SizovDBubbleSortSEQ::RunImpl() {
  const std::size_t n = data_.size();
  if (n <= 1) {
    return true;
  }

  bool swapped = true;
  while (swapped) {
    swapped = false;

    for (std::size_t i = 0; i + 1 < n; i += 2) {
      if (data_[i] > data_[i + 1]) {
        std::swap(data_[i], data_[i + 1]);
        swapped = true;
      }
    }

    for (std::size_t i = 1; i + 1 < n; i += 2) {
      if (data_[i] > data_[i + 1]) {
        std::swap(data_[i], data_[i + 1]);
        swapped = true;
      }
    }
  }

  return true;
}

bool SizovDBubbleSortSEQ::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace sizov_d_bubble_sort
