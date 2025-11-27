#include <gtest/gtest.h>

#include "ashihmin_d_sum_of_elem/common/include/common.hpp"
#include "ashihmin_d_sum_of_elem/mpi/include/ops_mpi.hpp"
#include "ashihmin_d_sum_of_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ashihmin_d_sum_of_elem {

class AshihminDElemVecSumPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 200000000;
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

TEST_P(AshihminDElemVecSumPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, AshihminDElemVecsSumMPI, AshihminDElemVecsSumSEQ>(
    PPC_SETTINGS_ashihmin_d_sum_of_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = AshihminDElemVecSumPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, AshihminDElemVecSumPerfTest, kGtestValues, kPerfTestName);

}  // namespace ashihmin_d_sum_of_elem
