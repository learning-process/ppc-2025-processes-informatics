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

class MinValueFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  InType input_data_{};
  OutType expected_output_{};

  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) {
      return output_data == expected_output_;
    }
    return true;
  }
};

namespace {

TEST_P(MinValueFuncTests, MinValueTestAllCases) {
  auto task_getter = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTaskGetter)>(GetParam());
  const auto &test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
  const auto &input_data = std::get<0>(test_params);
  const auto &test_name = std::get<2>(test_params);

  bool expect_validation_fail = (test_name.find("Invalid") != std::string::npos);

  if (expect_validation_fail) {
    if (test_name == "Invalid_Empty_Matrix" && !ppc::util::IsUnderMpirun()) {
      ASSERT_THROW(task_getter(input_data), std::invalid_argument);
    } else {
      auto task = task_getter(input_data);
      ASSERT_FALSE(task->Validation());
    }
  } else {
    auto task = task_getter(input_data);
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());
    ASSERT_TRUE(CheckTestOutputData(task->GetOutput()));
  }
}

const std::array<TestType, 5> kTestCases = {
    // Positive
    std::make_tuple(InType{{1, 2, 3}, {4, 5, 6}}, OutType{1, 4}, "Small_Matrix_2x3"),
    std::make_tuple(InType{{-1, 0}, {10, 2}, {7}}, OutType{-1, 2, 7}, "Matrix_With_Negative_3x2"),
    std::make_tuple(InType{{8}}, OutType{8}, "Single_Element_Matrix_1x1"),
    // Negative
    std::make_tuple(InType{}, OutType{}, "Invalid_Empty_Matrix"),
    std::make_tuple(InType{{1, 2}, {}}, OutType{}, "Invalid_Empty_Row")};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<MinValueSEQ, InType>(kTestCases, PPC_SETTINGS_makovskiy_i_min_value_in_matrix_rows),
    ppc::util::AddFuncTask<MinValueMPI, InType>(kTestCases, PPC_SETTINGS_makovskiy_i_min_value_in_matrix_rows));

const auto kGtestValues = ppc::util::ExpandToValues(kTasks);
const auto kTestName = MinValueFuncTests::PrintFuncTestName<MinValueFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinValueAllTests, MinValueFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace makovskiy_i_min_value_in_matrix_rows
