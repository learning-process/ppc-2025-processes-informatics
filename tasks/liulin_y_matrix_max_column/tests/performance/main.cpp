#include <gtest/gtest.h>

#include "liulin_y_matrix_max_column/common/include/common.hpp"
#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"
#include "liulin_y_matrix_max_column/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace liulin_y_matrix_max_column {

class LiulinYMatrixMaxColumnPerfTests
    : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};
  OutType expected_output_{};

  void SetUp() override {
    // Создаём квадратную матрицу kCount_ × kCount_
    input_data_.assign(kCount_, std::vector<int>(kCount_));

    for (int i = 0; i < kCount_; i++) {
      for (int j = 0; j < kCount_; j++) {
        input_data_[i][j] = i + j;  // Простой детерминированный набор данных
      }
    }

    // Ожидаемый результат: максимум по каждому столбцу
    expected_output_.assign(kCount_, 0);
    for (int col = 0; col < kCount_; col++) {
      int mx = input_data_[0][col];
      for (int row = 1; row < kCount_; row++) {
        mx = std::max(mx, input_data_[row][col]);
      }
      expected_output_[col] = mx;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LiulinYMatrixMaxColumnPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LiulinYMatrixMaxColumnMPI,
                                LiulinYMatrixMaxColumnSEQ>(
        PPC_SETTINGS_liulin_y_matrix_max_column);

const auto kGtestValues =
    ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName =
    LiulinYMatrixMaxColumnPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests,
                         LiulinYMatrixMaxColumnPerfTests,
                         kGtestValues,
                         kPerfTestName);

}  // namespace liulin_y_matrix_max_column
