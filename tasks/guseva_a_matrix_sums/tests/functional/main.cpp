#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "guseva_a_matrix_sums/common/include/common.hpp"
#include "guseva_a_matrix_sums/mpi/include/ops_mpi.hpp"
#include "guseva_a_matrix_sums/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

#define EPSILON 10e-12

namespace guseva_a_matrix_sums {

static TestType GenerateRandomTest(const std::string &name) {
  std::random_device rd;
  std::mt19937_64 gen(rd());

  std::uniform_real_distribution<double> dist_double(0.0, 100.0);
  std::uniform_int_distribution<int> dist_int(100, 200);

  uint32_t rows = dist_int(gen);
  uint32_t columns = dist_int(gen);
  std::vector<double> matrix;
  matrix.reserve(static_cast<uint64_t>(rows) * columns);
  for (uint32_t i = 0; i < rows * columns; i++) {
    matrix.push_back(static_cast<double>(dist_int(gen)));
  }

  auto input_data = guseva_a_matrix_sums::InType(rows, columns, matrix);
  std::vector<double> expected(columns, 0);
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < columns; j++) {
      expected[j] += matrix[(i * columns) + j];
    }
  }
  return {name, input_data, expected};
}

class GusevaARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    expected_data_ = std::get<2>(params);
    auto [a, b, c] = input_data_;
    std::print(
        std::cout,
        "\n\n================================================\nTEST CASE INFO\n\nTest name: {}\nTest sizes: {}x{} {}\n",
        std::get<0>(params), a, b, c[0]);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool should_compare = true;
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        should_compare = false;
      }
    }

    if (should_compare) {
      if (output_data.size() != expected_data_.size()) {
        return false;
      }
      for (uint32_t i = 0; i < expected_data_.size(); i++) {
        if (std::abs(output_data[i] - expected_data_[i]) > EPSILON) {
          return false;
        }
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

TEST_P(GusevaARunFuncTestsProcesses, MatrixSumsByColumnsSEQ) {
  const auto *current_test = ::testing::UnitTest::GetInstance()->current_test_info();
  std::print(std::cout, "\n\nTESTCASE NAME: {}\n\n", current_test->name());
  std::print(std::cout, "\n\nTESTCASE SUIT: {}\n\n", current_test->test_suite_name());
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {GenerateRandomTest("random_1"), GenerateRandomTest("random_2"),
                                             GenerateRandomTest("random_3"), GenerateRandomTest("random_4"),
                                             GenerateRandomTest("random_5")/*, GenerateRandomTest("random_6"),
                                             GenerateRandomTest("random_7"), GenerateRandomTest("random_8"),
                                             GenerateRandomTest("random_9"), GenerateRandomTest("random_10")*/};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<guseva_a_matrix_sums::GusevaAMatrixSumsMPI, InType>(
    kTestParam, PPC_SETTINGS_guseva_a_matrix_sums));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = GusevaARunFuncTestsProcesses::PrintFuncTestName<GusevaARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(GusevaAMatrix, GusevaARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace guseva_a_matrix_sums
