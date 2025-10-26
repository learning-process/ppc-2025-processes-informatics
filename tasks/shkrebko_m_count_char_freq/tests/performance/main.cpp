#include <gtest/gtest.h>

#include "shkrebko_m_count_char_freq/common/include/common.hpp"
#include "shkrebko_m_count_char_freq/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_count_char_freq/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkrebko_m_count_char_freq {

class ShkrebkoMCountCharFreqPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ShkrebkoMCountCharFreqPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShkrebkoMCountCharFreqMPI, ShkrebkoMCountCharFreqSEQ>(PPC_SETTINGS_shkrebko_m_count_char_freq);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShkrebkoMCountCharFreqPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShkrebkoMCountCharFreqPerfTests, kGtestValues, kPerfTestName);

}  // namespace shkrebko_m_count_char_freq
