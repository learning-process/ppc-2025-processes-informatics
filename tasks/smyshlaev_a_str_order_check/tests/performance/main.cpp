#include <gtest/gtest.h>

#include "smyshlaev_a_str_order_check/common/include/common.hpp"
#include "smyshlaev_a_str_order_check/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_str_order_check/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace smyshlaev_a_str_order_check {

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
    ppc::util::MakeAllPerfTasks<InType, SmyshlaevAStrOrderCheckMPI, SmyshlaevAStrOrderCheckSEQ>(PPC_SETTINGS_smyshlaev_a_str_order_check);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace smyshlaev_a_str_order_check
