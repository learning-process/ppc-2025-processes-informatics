#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"
#include "gutyansky_a_matrix_column_sum/mpi/include/ops_mpi.hpp"
#include "gutyansky_a_matrix_column_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gutyansky_a_matrix_column_sum {

class GutyanskyAMatrixColumnSumPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_.rows = kSize_;
    input_data_.cols = kSize_;
    input_data_.data.assign(kSize_ * kSize_, 1);
    output_data_.assign(kSize_, static_cast<int64_t>(kSize_));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  const size_t kSize_ = 8000;
  InType input_data_ = {};
  OutType output_data_;
};

TEST_P(GutyanskyAMatrixColumnSumPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GutyanskyAMatrixColumnSumMPI, GutyanskyAMatrixColumnSumSEQ>(
        PPC_SETTINGS_gutyansky_a_matrix_column_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GutyanskyAMatrixColumnSumPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GutyanskyAMatrixColumnSumPerfTest, kGtestValues, kPerfTestName);

}  // namespace gutyansky_a_matrix_column_sum
