#include "gusev_d_radix_double/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gusev_d_radix_double/common/include/common.hpp"

namespace gusev_d_radix_double {

namespace {

void CountSortPass(const uint64_t *source, uint64_t *dest, size_t n, int shift) {
  std::array<size_t, 256> count{};
  count.fill(0);

  for (size_t i = 0; i < n; ++i) {
    uint8_t byte = (source[i] >> shift) & 0xFF;
    count.at(byte)++;
  }

  size_t index = 0;
  for (size_t &i : count) {
    size_t tmp = i;
    i = index;
    index += tmp;
  }

  for (size_t i = 0; i < n; ++i) {
    uint8_t byte = (source[i] >> shift) & 0xFF;
    dest[count.at(byte)++] = source[i];
  }
}

}  // namespace

GusevDRadixDoubleSEQ::GusevDRadixDoubleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GusevDRadixDoubleSEQ::ValidationImpl() {
  return true;
}

bool GusevDRadixDoubleSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

void GusevDRadixDoubleSEQ::RadixSort(std::vector<double> &data) {
  if (data.size() < 2) {
    return;
  }

  size_t n = data.size();

  std::vector<uint64_t> raw_data(n);
  std::memcpy(raw_data.data(), data.data(), n * sizeof(double));

  for (size_t i = 0; i < n; ++i) {
    uint64_t mask = 0x8000000000000000ULL;
    if ((raw_data[i] & mask) != 0) {
      raw_data[i] = ~raw_data[i];
    } else {
      raw_data[i] |= mask;
    }
  }

  std::vector<uint64_t> buffer(n);
  uint64_t *source = raw_data.data();
  uint64_t *dest = buffer.data();

  for (int shift = 0; shift < 64; shift += 8) {
    CountSortPass(source, dest, n, shift);
    std::swap(source, dest);
  }

  for (size_t i = 0; i < n; ++i) {
    uint64_t u = source[i];
    uint64_t mask = 0x8000000000000000ULL;
    if ((u & mask) != 0) {
      u ^= mask;
    } else {
      u = ~u;
    }
    std::memcpy(&data[i], &u, sizeof(double));
  }
}

bool GusevDRadixDoubleSEQ::RunImpl() {
  RadixSort(GetOutput());
  return true;
}

bool GusevDRadixDoubleSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace gusev_d_radix_double
