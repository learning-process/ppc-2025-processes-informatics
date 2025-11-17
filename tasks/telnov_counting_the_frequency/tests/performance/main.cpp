#include <gtest/gtest.h>

#include "telnov_counting_the_frequency/common/include/common.hpp"
#include "telnov_counting_the_frequency/mpi/include/ops_mpi.hpp"
#include "telnov_counting_the_frequency/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace telnov_counting_the_frequency {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TelnovCountingTheFrequencyMPI, TelnovCountingTheFrequencySEQ>(PPC_SETTINGS_telnov_counting_the_frequency);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace telnov_counting_the_frequency
