#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "rozenberg_a_matrix_column_sum/common/include/common.hpp"
#include "rozenberg_a_matrix_column_sum/mpi/include/ops_mpi.hpp"
#include "rozenberg_a_matrix_column_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace rozenberg_a_matrix_column_sum {

class RozenbergAMatrixColumnSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType output_data_;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_rozenberg_a_matrix_column_sum, "perf_test.txt");
    std::ifstream file(abs_path);

    if (file.is_open()) {
      int rows = 0;
      int columns = 0;
      file >> rows >> columns;

      InType input_data(rows, std::vector<int>(columns));
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
          file >> input_data[i][j];
        }
      }

      OutType output_data(columns);
      for (int i = 0; i < columns; i++) {
        file >> output_data[i];
      }
      input_data_ = input_data;
      output_data_ = output_data;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RozenbergAMatrixColumnSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RozenbergAMatrixColumnSumMPI, RozenbergAMatrixColumnSumSEQ>(
        PPC_SETTINGS_rozenberg_a_matrix_column_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RozenbergAMatrixColumnSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RozenbergAMatrixColumnSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace rozenberg_a_matrix_column_sum
