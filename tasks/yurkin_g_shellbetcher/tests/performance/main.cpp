#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <ranges>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "yurkin_g_shellbetcher/common/include/common.hpp"
#include "yurkin_g_shellbetcher/mpi/include/ops_mpi.hpp"
#include "yurkin_g_shellbetcher/seq/include/ops_seq.hpp"

namespace yurkin_g_shellbetcher {

static std::int64_t ComputeExpectedChecksumSeq(int n) {
  std::vector<int> data;
  data.reserve(static_cast<std::size_t>(n));
  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);
  for (int i = 0; i < n; ++i) {
    data.push_back(dist(rng));
  }

  auto shell_sort_local = [](std::vector<int> &a) {
    std::size_t n_local = a.size();
    std::size_t gap = 1;
    while (gap < n_local / 3) {
      gap = (gap * 3) + 1;
    }
    while (gap > 0) {
      for (std::size_t i = gap; i < n_local; ++i) {
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
  };

  shell_sort_local(data);
  std::vector<int> left;
  std::vector<int> right;
  std::vector<int> merged;
  auto mid = data.size() / 2;
  left.assign(data.begin(), data.begin() + static_cast<std::vector<int>::difference_type>(mid));
  right.assign(data.begin() + static_cast<std::vector<int>::difference_type>(mid), data.end());
  merged.resize(left.size() + right.size());
  std::ranges::merge(left, right, merged.begin());
  for (int phase = 0; phase < 2; ++phase) {
    auto start = static_cast<std::size_t>(phase);
    for (std::size_t i = start; i + 1 < merged.size(); i += 2) {
      if (merged[i] > merged[i + 1]) {
        std::swap(merged[i], merged[i + 1]);
      }
    }
  }
  shell_sort_local(merged);
  std::int64_t checksum = 0;
  for (int v : merged) {
    checksum += static_cast<std::int64_t>(v);
  }
  return checksum & 0x7FFFFFFF;
}

class YurkinGShellBetcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_ = 0;

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::int64_t expected = ComputeExpectedChecksumSeq(static_cast<int>(input_data_));
    return static_cast<std::int64_t>(output_data) == expected;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(YurkinGShellBetcherPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YurkinGShellBetcherMPI, YurkinGShellBetcherSEQ>(
    PPC_SETTINGS_yurkin_g_shellbetcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, YurkinGShellBetcherPerfTests, kGtestValues,
                         YurkinGShellBetcherPerfTests::CustomPerfTestName);

}  // namespace yurkin_g_shellbetcher
