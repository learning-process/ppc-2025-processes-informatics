#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstdint>
#include <random>
#include <string>

#include "guseva_a_matrix_sums/common/include/common.hpp"
#include "guseva_a_matrix_sums/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

#define EPSILON 10e-12

namespace guseva_a_matrix_sums {

class GusevaARunFuncTestsProcessesSEQ : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "random_" + test_param;
  }

 protected:
  void SetUp() override {
    std::random_device rd;
    std::mt19937_64 gen(rd());

    std::uniform_real_distribution<double> dist_double(0.0, 100.0);
    std::uniform_int_distribution<int> dist_int(1, 300);

    uint32_t rows = dist_int(gen);
    uint32_t columns = dist_int(gen);
    std::vector<double> matrix;
    matrix.reserve(static_cast<uint64_t>(rows) * columns);
    for (uint32_t i = 0; i < rows * columns; i++) {
      matrix.push_back(dist_int(gen));
    }

    input_data_ = guseva_a_matrix_sums::InType(rows, columns, matrix);
    std::vector<double> expected(columns, 0);
    for (uint32_t i = 0; i < rows; i++) {
      for (uint32_t j = 0; j < columns; j++) {
        expected[j] += matrix[(i * columns) + j];
      }
    }
    expected_data_.assign(expected.begin(), expected.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // std::cout << "\n===================================\nINPUT\n";
    // int n = 0;
    // for (const auto &i : std::get<2>(input_data_)) {
    //   std::cout << i << ' ';
    //   n++;
    //   if (n % std::get<1>(input_data_) == 0) {
    //     std::cout << '\n';
    //   }
    // }

    // std::cout << "\nOUTPUT\n";
    // for (const auto &i : output_data) {
    //   std::cout << i << ' ';
    // }
    // std::cout << "\n\nEXPECTED\n";
    // for (const auto &i : expected_data_) {
    //   std::cout << i << ' ';
    // }

    // std::cout << "\n===================================\n";
    // if (output_data.size() != expected_data_.size()) {
    //   return false;
    // }
    for (uint32_t i = 0; i < expected_data_.size(); i++) {
      if (std::abs(output_data[i] - expected_data_[i]) > EPSILON) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_data_;
};

namespace {

TEST_P(GusevaARunFuncTestsProcessesSEQ, MatrixSumsByColumns) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};

const auto kTestTasksList =
    std::tuple_cat(/*ppc::util::AddFuncTask<GusevaARunFuncTestsProcessesMPI, InType>(kTestParam,
                      PPC_SETTINGS_example_processes),*/
                   ppc::util::AddFuncTask<guseva_a_matrix_sums::GusevaAMatrixSumsSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GusevaARunFuncTestsProcessesSEQ::PrintFuncTestName<GusevaARunFuncTestsProcessesSEQ>;

INSTANTIATE_TEST_SUITE_P(GusevaAMatrix, GusevaARunFuncTestsProcessesSEQ, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace guseva_a_matrix_sums
