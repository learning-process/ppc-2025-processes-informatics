#include <gtest/gtest.h>

#include "ovsyannikov_n_num_mistm_in_two_str/common/include/common.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_num_mistm_in_two_str/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

class OvsyannikovNRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(OvsyannikovNRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OvsyannikovNNumMistmInTwoStrMPI, OvsyannikovNNumMistmInTwoStrSEQ>(PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OvsyannikovNRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, OvsyannikovNRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace ovsyannikov_n_num_mistm_in_two_str
