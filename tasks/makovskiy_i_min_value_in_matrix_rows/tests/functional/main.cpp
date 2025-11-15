#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

// Positive
class MinValueRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &input = std::get<0>(test_param);
    const std::size_t rows = input.size();
    const std::size_t cols = input.empty() ? 0 : input.front().size();
    return "r" + std::to_string(rows) + "x" + std::to_string(cols);
  }

 protected:
  InType GetTestInputData() final {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const auto &expected = std::get<1>(params);
    return output_data == expected;
  }
};

TEST_P(MinValueRunFuncTests, MinPerRow) {
  ExecuteTest(GetParam());
}

namespace {

const auto kTestCases = std::array<TestType, 4>{
    TestType{InType{{1, 2, 3}, {4, 5, 6}}, OutType{1, 4}},
    TestType{InType{{-1, 0}, {10, 2}, {7}}, OutType{-1, 2, 7}},
    TestType{InType{{5, 5, 5}}, OutType{5}},
    TestType{InType{{8}}, OutType{8}},
};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<MinValueSEQ, InType>(kTestCases, PPC_SETTINGS_makovskiy_i_min_value_in_matrix_rows),
    ppc::util::AddFuncTask<MinValueMPI, InType>(kTestCases, PPC_SETTINGS_makovskiy_i_min_value_in_matrix_rows));

const auto kGtestValues = ppc::util::ExpandToValues(kTasks);
const auto kPerfTestName = MinValueRunFuncTests::PrintFuncTestName<MinValueRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinValuePositiveTests, MinValueRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

// Negative
TEST(MinValueNegativeTests, RejectsEmptyMatrix) {
  InType empty_input = {};

  ASSERT_THROW(MinValueSEQ task(empty_input), std::invalid_argument);

  if (ppc::util::IsUnderMpirun()) {
    auto task_mpi = std::make_shared<MinValueMPI>(empty_input);
    ASSERT_FALSE(task_mpi->Validation());
  }
}

TEST(MinValueNegativeTests, RejectsMatrixWithEmptyRow) {
  InType invalid_input = {{1, 2, 3}, {}};

  auto task_seq = std::make_shared<MinValueSEQ>(invalid_input);
  ASSERT_FALSE(task_seq->Validation());

  if (ppc::util::IsUnderMpirun()) {
    auto task_mpi = std::make_shared<MinValueMPI>(invalid_input);
    ASSERT_FALSE(task_mpi->Validation());
  }
}

}  // namespace makovskiy_i_min_value_in_matrix_rows
