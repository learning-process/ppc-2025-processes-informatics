#include <gtest/gtest.h>

#include "votincev_d_alternating_values/common/include/common.hpp"
#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"
#include "votincev_d_alternating_values/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatigValuesRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(VotincevDAlternatigValuesRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VotincevDAlternatingValuesMPI, VotincevDAlternatingValuesSEQ>(PPC_SETTINGS_votincev_d_alternating_values);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VotincevDAlternatigValuesRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VotincevDAlternatigValuesRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace votincev_d_alternating_values
