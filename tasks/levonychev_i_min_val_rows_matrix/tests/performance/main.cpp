#include <gtest/gtest.h>

#include "levonychev_i_min_val_rows_matrix/common/include/common.hpp"
#include "levonychev_i_min_val_rows_matrix/mpi/include/ops_mpi.hpp"
#include "levonychev_i_min_val_rows_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace levonychev_i_min_val_rows_matrix {

class LevonychevIMinValRowsMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(LevonychevIMinValRowsMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LevonychevIMinValRowsMatrixMPI, LevonychevIMinValRowsMatrixSEQ>(PPC_SETTINGS_levonychev_i_min_val_rows_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LevonychevIMinValRowsMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LevonychevIMinValRowsMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace levonychev_i_min_val_rows_matrix
