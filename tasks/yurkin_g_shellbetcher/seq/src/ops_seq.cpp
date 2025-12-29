#include "yurkin_g_shellbetcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "util/include/util.hpp"
#include "yurkin_g_shellbetcher/common/include/common.hpp"

namespace yurkin_g_shellbetcher {

YurkinGShellBetcherSEQ::YurkinGShellBetcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGShellBetcherSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool YurkinGShellBetcherSEQ::PreProcessingImpl() {
  return GetInput() > 0;
}

static void shell_sort(std::vector<int> &a) {
  size_t n = a.size();
  size_t gap = 1;
  while (gap < n / 3) {
    gap = gap * 3 + 1;
  }
  while (gap > 0) {
    for (size_t i = gap; i < n; ++i) {
      int tmp = a[i];
      size_t j = i;
      while (j >= gap && a[j - gap] > tmp) {
        a[j] = a[j - gap];
        j -= gap;
      }
      a[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}

static void odd_even_batcher_merge(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &out) {
  out.resize(a.size() + b.size());
  std::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
  for (int phase = 0; phase < 2; ++phase) {
    size_t start = static_cast<size_t>(phase);
    for (size_t i = start; i + 1 < out.size(); i += 2) {
      if (out[i] > out[i + 1]) {
        std::swap(out[i], out[i + 1]);
      }
    }
  }
}

bool YurkinGShellBetcherSEQ::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    return false;
  }

  std::vector<int> data;
  data.reserve(static_cast<size_t>(n));
  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);
  for (InType i = 0; i < n; ++i) {
    data.push_back(dist(rng));
  }

  shell_sort(data);

  std::vector<int> left, right, merged;
  size_t mid = data.size() / 2;
  left.assign(data.begin(), data.begin() + mid);
  right.assign(data.begin() + mid, data.end());
  odd_even_batcher_merge(left, right, merged);

  shell_sort(merged);

  long long checksum = 0;
  for (int v : merged) {
    checksum += v;
  }

  GetOutput() = static_cast<OutType>(checksum & 0x7FFFFFFF);
  return true;
}

bool YurkinGShellBetcherSEQ::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace  yurkin_g_shellbetcher
