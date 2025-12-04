#include <gtest/gtest.h>

#include "levonychev_i_mult_matrix_vec/common/include/common.hpp"
#include "levonychev_i_mult_matrix_vec/mpi/include/ops_mpi.hpp"
#include "levonychev_i_mult_matrix_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace levonychev_i_mult_matrix_vec {

class LevonychevIMultMatrixVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data >= 0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LevonychevIMultMatrixVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LevonychevIMultMatrixVecMPI, LevonychevIMultMatrixVecSEQ>(PPC_SETTINGS_levonychev_i_mult_matrix_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LevonychevIMultMatrixVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LevonychevIMultMatrixVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace levonychev_i_mult_matrix_vec
