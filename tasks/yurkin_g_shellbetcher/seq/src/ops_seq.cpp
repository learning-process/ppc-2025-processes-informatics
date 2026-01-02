#include "yurkin_g_shellbetcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "yurkin_g_shellbetcher/common/include/common.hpp"

namespace yurkin_g_shellbetcher {
namespace {

void ShellSort(std::vector<int> &a) {
  const std::size_t n = a.size();
  if (n < 2) {
    return;
  }
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

void CompareExchange(std::vector<int> &arr, int i, int j, bool ascending) {
  if (ascending) {
    if (arr[i] > arr[j]) {
      std::swap(arr[i], arr[j]);
    }
  } else {
    if (arr[i] < arr[j]) {
      std::swap(arr[i], arr[j]);
    }
  }
}

void BatcherOddEvenNetwork(std::vector<int> &arr, int length) {
  for (int p = 1; p < length; p <<= 1) {
    for (int q = p; q > 0; q >>= 1) {
      for (int i = 0; i < length; ++i) {
        int j = i ^ q;
        if (j > i) {
          bool ascending = ((i & p) == 0);
          CompareExchange(arr, i, j, ascending);
        }
      }
    }
  }
}

void BatcherMerge(const std::vector<int> &left, const std::vector<int> &right, std::vector<int> &out) {
  const std::size_t orig_n = left.size() + right.size();
  out.clear();
  out.reserve(orig_n);
  out.insert(out.end(), left.begin(), left.end());
  out.insert(out.end(), right.begin(), right.end());
  if (orig_n == 0) {
    return;
  }
  std::size_t pow2 = 1;
  while (pow2 < orig_n) {
    pow2 <<= 1;
  }
  const int sentinel = std::numeric_limits<int>::max();
  out.resize(pow2, sentinel);
  BatcherOddEvenNetwork(out, static_cast<int>(pow2));
  out.resize(orig_n);
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

  const std::size_t mid = data.size() / 2;
  std::vector<int> left;
  std::vector<int> right;
  left.assign(data.begin(), data.begin() + static_cast<std::vector<int>::difference_type>(mid));
  right.assign(data.begin() + static_cast<std::vector<int>::difference_type>(mid), data.end());

  std::vector<int> merged;
  BatcherMerge(left, right, merged);

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
