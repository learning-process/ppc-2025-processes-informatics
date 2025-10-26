#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"
#include "votincev_d_alternating_values/mpi/include/ops_mpi.hpp"
#include "votincev_d_alternating_values/seq/include/ops_seq.hpp"

namespace votincev_d_alternating_values {

class VotincevDAlternatigValuesRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};
  const int kCount_ = 100;

  void SetUp() override {
    int sz = kCount_ + 1;
    std::vector<double> v;
    int swapper = 1;
    for (int i = 0; i < sz; i++) {
      v.push_back(i * swapper);  // 0 -1 2 -3 4 -5...
      swapper *= -1;
    }

    input_data_ = v;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == kCount_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VotincevDAlternatingValuesMPI, VotincevDAlternatingValuesSEQ>(
        PPC_SETTINGS_votincev_d_alternating_values);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VotincevDAlternatigValuesRunPerfTestsProcesses::CustomPerfTestName;

TEST_P(VotincevDAlternatigValuesRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunPerf, VotincevDAlternatigValuesRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace votincev_d_alternating_values
