#include <gtest/gtest.h>

#include "smyshlaev_a_mat_mul/common/include/common.hpp"
#include "smyshlaev_a_mat_mul/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_mat_mul/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace smyshlaev_a_mat_mul {

class SmyshlaevAMatMulRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(SmyshlaevAMatMulRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SmyshlaevAMatMulMPI, SmyshlaevAMatMulSEQ>(PPC_SETTINGS_smyshlaev_a_mat_mul);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SmyshlaevAMatMulRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SmyshlaevAMatMulRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace smyshlaev_a_mat_mul
