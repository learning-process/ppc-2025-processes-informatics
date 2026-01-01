#include "akimov_i_radix_sort_double_batcher_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "akimov_i_radix_sort_double_batcher_merge/common/include/common.hpp"

namespace akimov_i_radix_sort_double_batcher_merge {

AkimovIRadixBatcherSortSEQ::AkimovIRadixBatcherSortSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool AkimovIRadixBatcherSortSEQ::ValidationImpl() {
  return true;
}
bool AkimovIRadixBatcherSortSEQ::PreProcessingImpl() {
  return true;
}
bool AkimovIRadixBatcherSortSEQ::PostProcessingImpl() {
  return true;
}

uint64_t AkimovIRadixBatcherSortSEQ::packDouble(double v) noexcept {
  uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  if (bits & (1ULL << 63)) {
    bits = ~bits;
  } else {
    bits ^= (1ULL << 63);
  }
  return bits;
}

double AkimovIRadixBatcherSortSEQ::unpackDouble(uint64_t k) noexcept {
  if (k & (1ULL << 63)) {
    k ^= (1ULL << 63);
  } else {
    k = ~k;
  }
  double v = 0.0;
  std::memcpy(&v, &k, sizeof(v));
  return v;
}

void AkimovIRadixBatcherSortSEQ::lsdRadixSort(std::vector<double> &arr) {
  const std::size_t n = arr.size();
  if (n < 2) {
    return;
  }

  constexpr int BITS = 8;
  constexpr int BUCKETS = 1 << BITS;
  constexpr int PASSES = (int)((sizeof(uint64_t) * 8) / BITS);

  std::vector<uint64_t> keys(n), tmpK(n);
  std::vector<double> tmpV(n);

  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = packDouble(arr[i]);
  }

  for (int pass = 0; pass < PASSES; ++pass) {
    int shift = pass * BITS;
    std::vector<std::size_t> cnt(BUCKETS + 1, 0);

    for (std::size_t i = 0; i < n; ++i) {
      std::size_t d = (keys[i] >> shift) & (BUCKETS - 1);
      ++cnt[d + 1];
    }
    for (int i = 0; i < BUCKETS; ++i) {
      cnt[i + 1] += cnt[i];
    }

    for (std::size_t i = 0; i < n; ++i) {
      std::size_t d = (keys[i] >> shift) & (BUCKETS - 1);
      std::size_t pos = cnt[d]++;
      tmpK[pos] = keys[i];
      tmpV[pos] = arr[i];
    }

    keys.swap(tmpK);
    arr.swap(tmpV);
  }

  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = unpackDouble(keys[i]);
  }
}

bool AkimovIRadixBatcherSortSEQ::RunImpl() {
  std::vector<double> data = GetInput();
  lsdRadixSort(data);
  GetOutput() = data;
  return true;
}

}  // namespace akimov_i_radix_sort_double_batcher_merge
