#include "gusev_d_radix_double/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "gusev_d_radix_double/common/include/common.hpp"

namespace gusev_d_radix_double {

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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void GusevDRadixDoubleSEQ::RadixSort(std::vector<double> &data) {
  if (data.size() < 2) {
    return;
  }

  size_t n = data.size();
  auto *ptr = reinterpret_cast<uint64_t *>(data.data());  // NOLINT

  for (size_t i = 0; i < n; ++i) {
    uint64_t u = ptr[i];  // NOLINT
    if ((u & 0x8000000000000000ULL) != 0) {
      ptr[i] = ~u;  // NOLINT
    } else {
      ptr[i] |= 0x8000000000000000ULL;  // NOLINT
    }
  }

  std::vector<uint64_t> buffer(n);
  uint64_t *source = ptr;
  uint64_t *dest = buffer.data();

  for (int shift = 0; shift < 64; shift += 8) {
    std::array<size_t, 256> count{};
    count.fill(0);

    for (size_t i = 0; i < n; ++i) {
      uint8_t byte = (source[i] >> shift) & 0xFF;  // NOLINT
      count[byte]++;                               // NOLINT
    }

    size_t index = 0;
    for (size_t &i : count) {
      size_t tmp = i;
      i = index;
      index += tmp;
    }

    for (size_t i = 0; i < n; ++i) {
      uint8_t byte = (source[i] >> shift) & 0xFF;  // NOLINT
      dest[count[byte]++] = source[i];             // NOLINT
    }

    std::swap(source, dest);
  }

  if (source != ptr) {
    std::memcpy(ptr, source, n * sizeof(uint64_t));
  }

  for (size_t i = 0; i < n; ++i) {
    uint64_t u = ptr[i];  // NOLINT
    if ((u & 0x8000000000000000ULL) != 0) {
      u ^= 0x8000000000000000ULL;
    } else {
      u = ~u;
    }
    ptr[i] = u;  // NOLINT
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
