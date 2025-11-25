#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <string>

#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"
#include "liulin_y_matrix_max_column/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace liulin_y_matrix_max_column {

class LiulinYMatrixMaxColumnPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  struct TestConfig {
    int rows;
    int cols;
    std::string name;
  };

  static TestConfig GetTestConfig(const std::string &test_name) {
    if (test_name.find("small") != std::string::npos) {
      return {100, 100, "small"};
    }
    if (test_name.find("medium") != std::string::npos) {
      return {1000, 1000, "medium"};
    }
    if (test_name.find("large") != std::string::npos) {
      return {10000, 10000, "large"};
    }
    if (test_name.find("xlarge") != std::string::npos) {
      return {20000, 20000, "xlarge"};
    }
    if (test_name.find("tall") != std::string::npos) {
      return {100, 5000, "tall"};
    }
    if (test_name.find("wide") != std::string::npos) {
      return {5000, 100, "wide"};
    }
    return {100, 100, "small"};
  }

  void SetUp() override {
    const auto &test_param = GetParam();
    const std::string &test_name = std::get<1>(test_param);

    TestConfig config = GetTestConfig(test_name);
    rows_ = config.rows;
    cols_ = config.cols;

    GenerateTestData();
  }

  void GenerateTestData() {
    input_data_.assign(rows_, std::vector<int>(cols_));

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        input_data_[i][j] = dist(gen);
      }
    }

    expected_output_.assign(cols_, std::numeric_limits<int>::min());
    for (int col = 0; col < cols_; col++) {
      for (int row = 0; row < rows_; row++) {
        if (input_data_[row][col] > expected_output_[col]) {
          expected_output_[col] = input_data_[row][col];
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  int rows_ = 0;
  int cols_ = 0;
  InType input_data_{};
  OutType expected_output_{};
};

TEST_P(LiulinYMatrixMaxColumnPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LiulinYMatrixMaxColumnMPI, LiulinYMatrixMaxColumnSEQ>(
    PPC_SETTINGS_liulin_y_matrix_max_column);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LiulinYMatrixMaxColumnPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LiulinYMatrixMaxColumnPerfTests, kGtestValues, kPerfTestName);

}  // namespace liulin_y_matrix_max_column
