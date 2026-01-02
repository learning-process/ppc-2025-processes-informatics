#include "gusev_d_radix_double/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstring>
#include <vector>
#include <cstdint>

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

void GusevDRadixDoubleSEQ::RadixSort(std::vector<double>& data) {
  if (data.size() < 2) return;

  size_t n = data.size();
  std::vector<uint64_t> raw_data(n);
  
  for (size_t i = 0; i < n; ++i) {
    uint64_t u;
    std::memcpy(&u, &data[i], sizeof(double));
    if ((u & 0x8000000000000000ULL)) {
      u = ~u;
    } else {
      u |= 0x8000000000000000ULL;
    }
    raw_data[i] = u;
  }

  std::vector<uint64_t> buffer(n);
  for (int shift = 0; shift < 64; shift += 8) {
    size_t count[256] = {0};
    
    for (size_t i = 0; i < n; ++i) {
      uint8_t byte = (raw_data[i] >> shift) & 0xFF;
      count[byte]++;
    }
    
    size_t index = 0;
    for (int i = 0; i < 256; ++i) {
      size_t tmp = count[i];
      count[i] = index;
      index += tmp;
    }
    
    for (size_t i = 0; i < n; ++i) {
      uint8_t byte = (raw_data[i] >> shift) & 0xFF;
      buffer[count[byte]] = raw_data[i];
      count[byte]++;
    }
    
    raw_data = buffer;
  }

  for (size_t i = 0; i < n; ++i) {
    uint64_t u = raw_data[i];
    if ((u & 0x8000000000000000ULL)) {
      u ^= 0x8000000000000000ULL;
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