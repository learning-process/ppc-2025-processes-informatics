#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <fstream>

#include "gutyansky_a_matrix_column_sum/common/include/common.hpp"
#include "gutyansky_a_matrix_column_sum/mpi/include/ops_mpi.hpp"
#include "gutyansky_a_matrix_column_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace gutyansky_a_matrix_column_sum {

class GutyanskyAMatrixColumnSumFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<double> input_elements;
    std::vector<double> output_elements;

    // Read test data
    {
      std::string file_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam()) + ".txt";
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_gutyansky_a_matrix_column_sum, file_name);

      std::ifstream ifs(abs_path);

      if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open test file: " + file_name);
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

    input_data_ = { rows, cols, input_elements };
    output_data_ = { cols, output_elements };
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = { };
  OutType output_data_ = { };
};

namespace {

TEST_P(GutyanskyAMatrixColumnSumFuncTests, MatrixColumnSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {"test_1", "test_2", "test_3"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<GutyanskyAMatrixColumnSumMPI, InType>(
                                               kTestParam, PPC_SETTINGS_gutyansky_a_matrix_column_sum),
                                           ppc::util::AddFuncTask<GutyanskyAMatrixColumnSumSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_gutyansky_a_matrix_column_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GutyanskyAMatrixColumnSumFuncTests::PrintFuncTestName<GutyanskyAMatrixColumnSumFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, GutyanskyAMatrixColumnSumFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace gutyansky_a_matrix_column_sum
