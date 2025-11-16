#include <gtest/gtest.h>

#include "maslova_u_char_frequency_count/common/include/common.hpp"
#include "maslova_u_char_frequency_count/mpi/include/ops_mpi.hpp"
#include "maslova_u_char_frequency_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace maslova_u_char_frequency_count {

class MaslovaURunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(MaslovaURunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MaslovaUCharFrequencyCountMPI, MaslovaUCharFrequencyCountSEQ>(PPC_SETTINGS_maslova_u_char_frequency_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaslovaURunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MaslovaURunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace maslova_u_char_frequency_count
