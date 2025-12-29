#include "yurkin_g_shellbetcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

#include "yurkin_g_shellbetcher/common/include/common.hpp"

namespace yurkin_g_shellbetcher {
namespace {

void ShellSort(std::vector<int> &a) {
  std::size_t n = a.size();
  std::size_t gap = 1;
  while (gap < n / 3) {
    gap = (gap * 3) + 1;
  }
  while (gap > 0) {
    for (std::size_t i = gap; i < n; ++i) {
      int tmp = a[i];
      std::size_t j = i;
      while (j >= gap && a[j - gap] > tmp) {
        a[j] = a[j - gap];
        j -= gap;
      }
      a[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}

void OddEvenBatcherMerge(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &out) {
  out.resize(a.size() + b.size());
  std::ranges::merge(a, b, out.begin());
  for (int phase = 0; phase < 2; ++phase) {
    auto start = static_cast<std::size_t>(phase);
    for (std::size_t i = start; i + 1 < out.size(); i += 2) {
      if (out[i] > out[i + 1]) {
        std::swap(out[i], out[i + 1]);
      }
    }
  }
}

}  // namespace

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

bool YurkinGShellBetcherSEQ::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    return false;
  }

  std::vector<int> data;
  data.reserve(static_cast<std::size_t>(n));
  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);
  for (InType i = 0; i < n; ++i) {
    data.push_back(dist(rng));
  }

  ShellSort(data);

  std::vector<int> left;
  std::vector<int> right;
  std::vector<int> merged;
  auto mid = data.size() / 2;
  left.assign(data.begin(), data.begin() + static_cast<std::vector<int>::difference_type>(mid));
  right.assign(data.begin() + static_cast<std::vector<int>::difference_type>(mid), data.end());
  OddEvenBatcherMerge(left, right, merged);

  ShellSort(merged);

  std::int64_t checksum = 0;
  for (int v : merged) {
    checksum += static_cast<std::int64_t>(v);
  }

  GetOutput() = static_cast<OutType>(checksum & 0x7FFFFFFF);
  return true;
}

bool YurkinGShellBetcherSEQ::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace yurkin_g_shellbetcher
