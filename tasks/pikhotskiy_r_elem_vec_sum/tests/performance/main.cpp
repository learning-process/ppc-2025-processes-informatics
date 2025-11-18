#include <gtest/gtest.h>

#include "pikhotskiy_r_elem_vec_sum/common/include/common.hpp"
#include "pikhotskiy_r_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pikhotskiy_r_elem_vec_sum {

class PikhotskiyRElelmVecSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(PikhotskiyRElelmVecSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PikhotskiyRElemVecSumMPI, PikhotskiyRElemVecSumSEQ>(PPC_SETTINGS_pikhotskiy_r_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PikhotskiyRElelmVecSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PikhotskiyRElelmVecSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace pikhotskiy_r_elem_vec_sum
