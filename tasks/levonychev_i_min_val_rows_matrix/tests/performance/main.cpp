#include <gtest/gtest.h>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"
#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace levonychev_i_min_val_rows_matrix {

class LevonychevIMinValRowsMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int ROWS = 10000;
  const int COLS = 10000;
  InType input_data_;
  OutType expected_result_;
  void SetUp() override {
    std::vector<int> matrix(ROWS * COLS);
    for (int i = 0; i < COLS * ROWS; ++i) {
      matrix[i] = i;
    }
    input_data_ = std::make_tuple(std::move(matrix), ROWS, COLS);
    for (int i = 0; i < ROWS * COLS; i += COLS) {
      expected_result_.push_back(i);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LevonychevIMinValRowsMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LevonychevIMinValRowsMatrixMPI, LevonychevIMinValRowsMatrixSEQ>(
        PPC_SETTINGS_levonychev_i_min_val_rows_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LevonychevIMinValRowsMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LevonychevIMinValRowsMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace levonychev_i_min_val_rows_matrix
