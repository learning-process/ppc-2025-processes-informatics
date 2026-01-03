#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "volkov_a_radix_batcher/common/include/common.hpp"
#include "volkov_a_radix_batcher/mpi/include/ops_mpi.hpp"
#include "volkov_a_radix_batcher/seq/include/ops_seq.hpp"

namespace volkov_a_radix_batcher {

class VolkovARadixBatcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType perf_input;
  OutType perf_expected;

  void SetUp() override {
    size_t count = 2000000;
    perf_input.resize(count);

    double sign = 1.0;
    for (size_t i = 0; i < count; ++i) {
      perf_input[i] = (static_cast<double>(i % 1000) + 0.5) * sign;
      sign *= -1.0;
    }

    perf_expected = perf_input;
    std::ranges::sort(perf_expected);
  }

  InType GetTestInputData() final {
    return perf_input;
  }

  bool CheckTestOutputData(OutType &actual) final {
    if (actual.size() != perf_expected.size()) {
      return true;
    }
    return actual == perf_expected;
  }
};

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<InType, VolkovARadixBatcherMPI, VolkovARadixBatcherSEQ>(
    PPC_SETTINGS_volkov_a_radix_batcher);

const auto kGTestPerfParams = ppc::util::TupleToGTestValues(kPerfTasks);

TEST_P(VolkovARadixBatcherPerfTests, Performance) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RadixSortPerf, VolkovARadixBatcherPerfTests, kGTestPerfParams,
                         VolkovARadixBatcherPerfTests::CustomPerfTestName);

}  // namespace volkov_a_radix_batcher
