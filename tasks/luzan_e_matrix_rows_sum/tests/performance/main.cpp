#include <gtest/gtest.h>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"
#include "luzan_e_matrix_rows_sum/mpi/include/ops_mpi.hpp"
#include "luzan_e_matrix_rows_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luzan_e_matrix_rows_sum {

class LuzanEMatrixRowsSumpERFTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int HEIGHT = 50; // 300 000 000
  const int WIDTH = 3000000;
  InType input_data_{};

  void SetUp() override {
    std::tuple_element_t<0, InType> mat(HEIGHT * WIDTH);
    for (int elem = 0; elem < HEIGHT * WIDTH; elem++) {
      mat[elem] = elem - 42;
    }

    input_data_ = std::make_tuple(mat, HEIGHT, WIDTH);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> sum(HEIGHT);
    std::tuple_element_t<0, InType> mat = std::get<0>(input_data_);

    for (int row = 0; row < HEIGHT; row++) {
      for (int col = 0; col < WIDTH; col++) {
        sum[row] += mat[WIDTH * row + col];
      }
    }

    return (output_data == sum);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuzanEMatrixRowsSumpERFTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LuzanEMatrixRowsSumMPI, LuzanEMatrixRowsSumSEQ>(
    PPC_SETTINGS_luzan_e_matrix_rows_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuzanEMatrixRowsSumpERFTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuzanEMatrixRowsSumpERFTests, kGtestValues, kPerfTestName);

}  // namespace luzan_e_matrix_rows_sum
