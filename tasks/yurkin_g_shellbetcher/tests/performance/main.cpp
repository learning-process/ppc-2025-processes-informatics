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
  for (int i = 0; i < n; ++i) data.push_back(dist(rng));
  std::ranges::sort(data);
  std::int64_t checksum = 0;
  for (int v : data) checksum += static_cast<std::int64_t>(v);
  return checksum & 0x7FFFFFFF;
}

class YurkinGShellBetcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000;
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
