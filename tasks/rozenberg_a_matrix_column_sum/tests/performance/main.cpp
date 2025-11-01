#include <gtest/gtest.h>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"
#include "rozenberg_a_matrix_column_sum/mpi/include/ops_mpi.hpp"
#include "rozenberg_a_matrix_column_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rozenberg_a_matrix_column_sum {

class RozenbergAMatrixColumnSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(RozenbergAMatrixColumnSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RozenbergAMatrixColumnSumMPI, RozenbergAMatrixColumnSumSEQ>(PPC_SETTINGS_rozenberg_a_matrix_column_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RozenbergAMatrixColumnSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RozenbergAMatrixColumnSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace rozenberg_a_matrix_column_sum
