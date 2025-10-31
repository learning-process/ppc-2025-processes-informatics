#include <gtest/gtest.h>

#include "nesterov_a_elem_vec_sum/common/include/common.hpp"
#include "nesterov_a_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "nesterov_a_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nesterov_a_elem_vec_sum {

class NesterovAElemVecSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(NesterovAElemVecSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NesterovAElemVecSumMPI, NesterovAElemVecSumSEQ>(PPC_SETTINGS_nesterov_a_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NesterovAElemVecSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NesterovAElemVecSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace nesterov_a_elem_vec_sum
