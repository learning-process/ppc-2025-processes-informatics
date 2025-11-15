#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "makovskiy_i_min_value_in_matrix_rows/common/include/common.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_min_value_in_matrix_rows/seq/include/ops_seq.hpp"
#include "util/include/util.hpp"

namespace makovskiy_i_min_value_in_matrix_rows {

enum class TaskType { SEQ, MPI };
enum class TestExpectation { SUCCESS, FAIL_VALIDATION };

using UnifiedTestParam = std::tuple<TaskType, std::string, InType, OutType, TestExpectation>;

class MinValueAllTests : public ::testing::TestWithParam<UnifiedTestParam> {};

TEST_P(MinValueAllTests, AllCases) {
  auto task_type = std::get<0>(GetParam());
  auto input_data = std::get<2>(GetParam());
  auto expected_output = std::get<3>(GetParam());
  auto expectation = std::get<4>(GetParam());

  std::shared_ptr<BaseTask> task;
  switch (task_type) {
    case TaskType::SEQ:
      task = std::make_shared<MinValueSEQ>(input_data);
      break;
    case TaskType::MPI:
      task = std::make_shared<MinValueMPI>(input_data);
      break;
  }
  ASSERT_NE(task, nullptr);

  switch (expectation) {
    case TestExpectation::SUCCESS: {
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
      ASSERT_FALSE(task->Validation());
      break;
    }
  }
}

TEST(MinValueNegativeTests, SeqThrowsOnEmptyMatrix) {
  if (ppc::util::IsUnderMpirun()) {
    GTEST_SKIP() << "Skipping constructor exception test under MPI";
  }
  InType empty_input = {};
  ASSERT_THROW((void)std::make_shared<MinValueSEQ>(empty_input), std::invalid_argument);
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
        UnifiedTestParam{TaskType::SEQ, "SEQ/Positive/r2x3", {{1, 2, 3}, {4, 5, 6}}, {1, 4}, TestExpectation::SUCCESS},
        UnifiedTestParam{TaskType::MPI, "MPI/Positive/r2x3", {{1, 2, 3}, {4, 5, 6}}, {1, 4}, TestExpectation::SUCCESS},
        UnifiedTestParam{TaskType::SEQ,
                         "SEQ/Positive/NegativeValues",
                         {{-1, 0}, {10, 2}, {7}},
                         {-1, 2, 7},
                         TestExpectation::SUCCESS},
        UnifiedTestParam{TaskType::MPI,
                         "MPI/Positive/NegativeValues",
                         {{-1, 0}, {10, 2}, {7}},
                         {-1, 2, 7},
                         TestExpectation::SUCCESS},
        UnifiedTestParam{TaskType::SEQ, "SEQ/Positive/SingleValue", {{8}}, {8}, TestExpectation::SUCCESS},
        UnifiedTestParam{TaskType::MPI, "MPI/Positive/SingleValue", {{8}}, {8}, TestExpectation::SUCCESS},

        // Negative
        UnifiedTestParam{TaskType::MPI, "MPI/Negative/EmptyMatrix", {}, {}, TestExpectation::FAIL_VALIDATION},
        UnifiedTestParam{TaskType::SEQ, "SEQ/Negative/EmptyRow", {{1, 2}, {}}, {}, TestExpectation::FAIL_VALIDATION},
        UnifiedTestParam{TaskType::MPI, "MPI/Negative/EmptyRow", {{1, 2}, {}}, {}, TestExpectation::FAIL_VALIDATION}),
    GenerateTestName);

}  // namespace makovskiy_i_min_value_in_matrix_rows
