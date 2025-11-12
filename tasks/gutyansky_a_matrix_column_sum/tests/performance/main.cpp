#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"
#include "gutyansky_a_matrix_column_sum/mpi/include/ops_mpi.hpp"
#include "gutyansky_a_matrix_column_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_matrix_column_sum {

class GutyanskyAMatrixColumnSumPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<int64_t> input_elements;
    std::vector<int64_t> output_elements;

    // Read test data
    {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gutyansky_a_matrix_column_sum, "test_perf.txt");

      std::ifstream ifs(abs_path);

      if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open test file: test_perf.txt");
      }

      ifs >> rows >> cols;

      if (rows == 0 || cols == 0) {
        throw std::runtime_error("Both dimensions of matrix must be positive integers");
      }

      input_elements.resize(rows * cols);

      for (size_t i = 0; i < input_elements.size(); i++) {
        ifs >> input_elements[i];
      }

      output_elements.resize(cols);

      for (size_t i = 0; i < output_elements.size(); i++) {
        ifs >> output_elements[i];
      }
    }

    input_data_ = {.rows = rows, .cols = cols, .data = input_elements};
    output_data_ = output_elements;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = {};
  OutType output_data_ = {};
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
