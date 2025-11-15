#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/seq/include/ops_seq.hpp"
#include "util/include/util.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

enum class TaskType { SEQ, MPI };
enum class TestExpectation { SUCCESS, FAIL_VALIDATION, THROW_CONSTRUCTION };

using TestParam = std::tuple<TaskType, std::string, InType, OutType, TestExpectation>;

class MinValueFunctionalTests : public ::testing::TestWithParam<TestParam> {};

TEST_P(MinValueFunctionalTests, AllCases) {
  const auto &[task_type, test_name, input_data, expected_output, expectation] = GetParam();

  switch (expectation) {
    case TestExpectation::SUCCESS: {
      std::shared_ptr<BaseTask> task;
      if (task_type == TaskType::SEQ) {
        task = std::make_shared<MinValueSEQ>(input_data);
      } else {
        task = std::make_shared<MinValueMPI>(input_data);
      }

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
      std::shared_ptr<BaseTask> task;
      if (task_type == TaskType::SEQ) {
        task = std::make_shared<MinValueSEQ>(input_data);
      } else {
        task = std::make_shared<MinValueMPI>(input_data);
      }

      ASSERT_FALSE(task->Validation());
      task->PreProcessing();
      task->Run();
      task->PostProcessing();
      break;
    }
    case TestExpectation::THROW_CONSTRUCTION: {
      if (task_type == TaskType::SEQ && !ppc::util::IsUnderMpirun()) {
        ASSERT_THROW((MinValueSEQ(input_data)), std::invalid_argument);
      }
      break;
    }
  }
}

auto GenerateTestName = [](const ::testing::TestParamInfo<TestParam> &info) {
  std::string name = std::get<1>(info.param);
  TaskType type = std::get<0>(info.param);
  std::string type_str = (type == TaskType::SEQ) ? "SEQ_" : "MPI_";
  std::replace(name.begin(), name.end(), '/', '_');
  return type_str + name;
};

INSTANTIATE_TEST_SUITE_P(
    MinValueTests, MinValueFunctionalTests,
    ::testing::Values(
        // Positive
        TestParam{TaskType::SEQ, "Positive_2x3", {{1, 2, 3}, {4, 5, 6}}, {1, 4}, TestExpectation::SUCCESS},
        TestParam{TaskType::MPI, "Positive_2x3", {{1, 2, 3}, {4, 5, 6}}, {1, 4}, TestExpectation::SUCCESS},
        TestParam{
            TaskType::SEQ, "Positive_NegativeVals", {{-1, 0}, {10, 2}, {7}}, {-1, 2, 7}, TestExpectation::SUCCESS},
        TestParam{
            TaskType::MPI, "Positive_NegativeVals", {{-1, 0}, {10, 2}, {7}}, {-1, 2, 7}, TestExpectation::SUCCESS},
        TestParam{TaskType::SEQ, "Positive_Single", {{8}}, {8}, TestExpectation::SUCCESS},
        TestParam{TaskType::MPI, "Positive_Single", {{8}}, {8}, TestExpectation::SUCCESS},

        // Negative: THROW_CONSTRUCTION
        TestParam{TaskType::SEQ, "Invalid_EmptyMatrix", {}, {}, TestExpectation::THROW_CONSTRUCTION},

        // Negative: FAIL_VALIDATION
        TestParam{TaskType::MPI, "Invalid_EmptyMatrix", {}, {}, TestExpectation::FAIL_VALIDATION},
        TestParam{TaskType::SEQ, "Invalid_EmptyRow", {{1, 2}, {}}, {}, TestExpectation::FAIL_VALIDATION},
        TestParam{TaskType::MPI, "Invalid_EmptyRow", {{1, 2}, {}}, {}, TestExpectation::FAIL_VALIDATION}),
    GenerateTestName);

}  // namespace makovskiy_i_min_value_in_matrix_rows
