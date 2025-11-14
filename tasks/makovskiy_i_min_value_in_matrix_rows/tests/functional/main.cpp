#include <gtest/gtest.h>
#include <mpi.h>

#include <functional>
#include <stdexcept>
#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/seq/include/ops_seq.hpp"
#include "util/include/util.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

enum class TestExpectation { SUCCESS, FAIL_VALIDATION, THROW_CONSTRUCTION };

using UnifiedTestParam =
    std::tuple<std::function<std::shared_ptr<BaseTask>(const InType &)>, std::string, InType, OutType, TestExpectation>;

class MinValueAllTests : public ::testing::TestWithParam<UnifiedTestParam> {};

TEST_P(MinValueAllTests, AllCases) {
  auto task_factory = std::get<0>(GetParam());
  auto test_name = std::get<1>(GetParam());
  auto input_data = std::get<2>(GetParam());
  auto expected_output = std::get<3>(GetParam());
  auto expectation = std::get<4>(GetParam());

  switch (expectation) {
    case TestExpectation::SUCCESS: {
      auto task = task_factory(input_data);
      ASSERT_TRUE(task->Validation());
      ASSERT_TRUE(task->PreProcessing());
      ASSERT_TRUE(task->Run());
      ASSERT_TRUE(task->PostProcessing());

      if (ppc::util::IsUnderMpirun()) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
          ASSERT_EQ(task->GetOutput(), expected_output);
        }
      } else {
        ASSERT_EQ(task->GetOutput(), expected_output);
      }
      break;
    }
    case TestExpectation::FAIL_VALIDATION: {
      auto task = task_factory(input_data);
      ASSERT_FALSE(task->Validation());
      break;
    }
    case TestExpectation::THROW_CONSTRUCTION: {
      if (!ppc::util::IsUnderMpirun()) {
        ASSERT_THROW(task_factory(input_data), std::invalid_argument);
      }
      break;
    }
  }
}

auto GenerateTestName = [](const ::testing::TestParamInfo<UnifiedTestParam> &info) {
  std::string name = std::get<1>(info.param);
  std::replace(name.begin(), name.end(), '/', '_');
  return name;
};

INSTANTIATE_TEST_SUITE_P(
    MinValueTests, MinValueAllTests,
    ::testing::Values(
        // Positive
        UnifiedTestParam{[](const InType &in) {
                           return std::make_shared<MinValueSEQ>(in);
                         }, "SEQ/Positive/r2x3", {{1, 2, 3}, {4, 5, 6}}, {1, 4}, TestExpectation::SUCCESS},
        UnifiedTestParam{[](const InType &in) {
                           return std::make_shared<MinValueMPI>(in);
                         }, "MPI/Positive/r2x3", {{1, 2, 3}, {4, 5, 6}}, {1, 4}, TestExpectation::SUCCESS},
        UnifiedTestParam{
            [](const InType &in) {
              return std::make_shared<MinValueSEQ>(in);
            }, "SEQ/Positive/NegativeValues", {{-1, 0}, {10, 2}, {7}}, {-1, 2, 7}, TestExpectation::SUCCESS},
        UnifiedTestParam{
            [](const InType &in) {
              return std::make_shared<MinValueMPI>(in);
            }, "MPI/Positive/NegativeValues", {{-1, 0}, {10, 2}, {7}}, {-1, 2, 7}, TestExpectation::SUCCESS},
        UnifiedTestParam{[](const InType &in) {
                           return std::make_shared<MinValueSEQ>(in);
                         }, "SEQ/Positive/SingleValue", {{8}}, {8}, TestExpectation::SUCCESS},
        UnifiedTestParam{[](const InType &in) {
                           return std::make_shared<MinValueMPI>(in);
                         }, "MPI/Positive/SingleValue", {{8}}, {8}, TestExpectation::SUCCESS},

        // Negative
        UnifiedTestParam{[](const InType &in) {
                           return std::make_shared<MinValueSEQ>(in);
                         }, "SEQ/Negative/EmptyMatrix", {}, {}, TestExpectation::THROW_CONSTRUCTION},
        UnifiedTestParam{[](const InType &in) {
                           return std::make_shared<MinValueMPI>(in);
                         }, "MPI/Negative/EmptyMatrix", {}, {}, TestExpectation::FAIL_VALIDATION},
        UnifiedTestParam{[](const InType &in) {
                           return std::make_shared<MinValueSEQ>(in);
                         }, "SEQ/Negative/EmptyRow", {{1, 2}, {}}, {}, TestExpectation::FAIL_VALIDATION},
        UnifiedTestParam{[](const InType &in) {
                           return std::make_shared<MinValueMPI>(in);
                         }, "MPI/Negative/EmptyRow", {{1, 2}, {}}, {}, TestExpectation::FAIL_VALIDATION}),
    GenerateTestName);

}  // namespace makovskiy_i_min_value_in_matrix_rows
