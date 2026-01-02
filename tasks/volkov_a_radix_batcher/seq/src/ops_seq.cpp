#include "volkov_a_radix_batcher/seq/include/ops_seq.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "volkov_a_radix_batcher/common/include/common.hpp"

namespace volkov_a_radix_batcher {

VolkovARadixBatcherSEQ::VolkovARadixBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VolkovARadixBatcherSEQ::ValidationImpl() {
  return true;
}

bool VolkovARadixBatcherSEQ::PreProcessingImpl() {
  return true;
}

uint64_t VolkovARadixBatcherSEQ::DoubleToOrderedInt(double d) {
  uint64_t u = 0;
  std::memcpy(&u, &d, sizeof(d));
  uint64_t mask = (static_cast<uint64_t>(1) << 63);
  if ((u & mask) != 0) {
    return ~u;
  }
  return u | mask;
}

double VolkovARadixBatcherSEQ::OrderedIntToDouble(uint64_t k) {
  uint64_t mask = (static_cast<uint64_t>(1) << 63);
  if ((k & mask) != 0) {
    k &= ~mask;
  } else {
    k = ~k;
  }
  double d = 0.0;
  std::memcpy(&d, &k, sizeof(d));
  return d;
}

void VolkovARadixBatcherSEQ::RadixSortDouble(std::vector<double> &data) {
  if (data.empty()) {
    return;
  }

  std::vector<uint64_t> keys(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    keys[i] = DoubleToOrderedInt(data[i]);
  }

  std::vector<uint64_t> temp(data.size());
  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<size_t> counts(256, 0);
    for (uint64_t k : keys) {
      counts[(k >> shift) & 0xFF]++;
    }

    std::vector<size_t> positions(256);
    positions[0] = 0;
    for (int i = 1; i < 256; i++) {
      positions[i] = positions[i - 1] + counts[i - 1];
    }

    for (uint64_t k : keys) {
      temp[positions[(k >> shift) & 0xFF]++] = k;
    }
    keys = temp;
  }

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = OrderedIntToDouble(keys[i]);
  }
}

bool VolkovARadixBatcherSEQ::RunImpl() {
  auto data = GetInput();
  RadixSortDouble(data);
  GetOutput() = data;
  return true;
}

bool VolkovARadixBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_radix_batcher