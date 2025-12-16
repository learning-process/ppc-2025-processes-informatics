#include "egashin_k_radix_batcher_sort/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "egashin_k_radix_batcher_sort/common/include/common.hpp"

namespace egashin_k_radix_batcher_sort {

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TestTaskSEQ::ValidationImpl() { return true; }

bool TestTaskSEQ::PreProcessingImpl() { return true; }

bool TestTaskSEQ::PostProcessingImpl() { return true; }

uint64_t TestTaskSEQ::DoubleToSortable(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(double));
  // For positive numbers: flip the sign bit
  // For negative numbers: flip all bits
  if ((bits & (1ULL << 63)) != 0) {
    bits = ~bits;
  } else {
    bits ^= (1ULL << 63);
  }
  return bits;
}

double TestTaskSEQ::SortableToDouble(uint64_t bits) {
  // Reverse the transformation
  if ((bits & (1ULL << 63)) != 0) {
    bits ^= (1ULL << 63);
  } else {
    bits = ~bits;
  }
  double value = 0;
  std::memcpy(&value, &bits, sizeof(double));
  return value;
}

void TestTaskSEQ::RadixSort(std::vector<double> &arr) {
  if (arr.size() <= 1) {
    return;
  }

  const int kBitsPerPass = 8;
  const int kNumBuckets = 256;
  const int kNumPasses = sizeof(uint64_t) * 8 / kBitsPerPass;

  std::size_t n = arr.size();
  std::vector<uint64_t> keys(n);
  std::vector<uint64_t> temp_keys(n);
  std::vector<double> temp_values(n);

  // Convert doubles to sortable integers
  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = DoubleToSortable(arr[i]);
  }

  // LSD radix sort
  for (int pass = 0; pass < kNumPasses; ++pass) {
    int shift = pass * kBitsPerPass;

    // Count occurrences
    std::vector<std::size_t> count(kNumBuckets + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t digit = (keys[i] >> shift) & 0xFF;
      count[digit + 1]++;
    }

    // Compute prefix sums
    for (int i = 0; i < kNumBuckets; ++i) {
      count[i + 1] += count[i];
    }

    // Distribute elements
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t digit = (keys[i] >> shift) & 0xFF;
      std::size_t pos = count[digit]++;
      temp_keys[pos] = keys[i];
      temp_values[pos] = arr[i];
    }

    std::swap(keys, temp_keys);
    std::swap(arr, temp_values);
  }

  // Convert back to doubles
  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = SortableToDouble(keys[i]);
  }
}

bool TestTaskSEQ::RunImpl() {
  std::vector<double> data = GetInput();
  RadixSort(data);
  GetOutput() = data;
  return true;
}

}  // namespace egashin_k_radix_batcher_sort

