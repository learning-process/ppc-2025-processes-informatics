#include <gtest/gtest.h>

#include "kurpiakov_a_shellsort/common/include/common.hpp"
#include "kurpiakov_a_shellsort/mpi/include/ops_mpi.hpp"
#include "kurpiakov_a_shellsort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kurpiakov_a_shellsort {

class KurpiakovARunPerfTestProcesses3 : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(KurpiakovARunPerfTestProcesses3, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KurpiakovAShellsortMPI, KurpiakovAShellsortSEQ>(PPC_SETTINGS_kurpiakov_a_shellsort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KurpiakovARunPerfTestProcesses3::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KurpiakovARunPerfTestProcesses3, kGtestValues, kPerfTestName);

}  // namespace kurpiakov_a_shellsort
