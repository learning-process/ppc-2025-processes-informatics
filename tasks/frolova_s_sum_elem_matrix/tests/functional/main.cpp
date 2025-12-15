#include <gtest/gtest.h>

#include <array>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"
#include "frolova_s_sum_elem_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace frolova_s_sum_elem_matrix {

class FrolovaSSumElemMatrixRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int rows = std::get<0>(test_param);
    int cols = std::get<1>(test_param);
    const std::string &label = std::get<2>(test_param);
    return std::to_string(rows) + "x" + std::to_string(cols) + "_" + label;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int rows = std::get<0>(params);
    int cols = std::get<1>(params);
    const std::string &label = std::get<2>(params);

    // Создаем только непустые матрицы
    if (rows > 0 && cols > 0) {
      matrix_.resize(rows);
      for (auto &row : matrix_) {
        row.resize(cols, 1);
      }
      expected_sum_ = static_cast<OutType>(rows) * cols;
    } else if (label == "jagged_matrix") {
      matrix_ = {{1, 2, 3}, {4, 5}};
      expected_sum_ = 15;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool result = (output_data == expected_sum_);
    if (!result) {
      std::cerr << "DEBUG: Test failed! Expected: " << expected_sum_ << ", Got: " << output_data << '\n';
    }
    return result;
  }

  InType GetTestInputData() final {
    return matrix_;
  }

 private:
  InType matrix_;
  OutType expected_sum_{0};
};

namespace {

// Общий тест для SEQ и MPI
TEST_P(FrolovaSSumElemMatrixRunFuncTests, SumElementsInMatrix) {
  auto param = GetParam();
  InType input_matrix = GetTestInputData();

  // SEQ: конструктор сразу принимает матрицу
  FrolovaSSumElemMatrixSEQ task_seq(input_matrix);
  EXPECT_TRUE(task_seq.Validation());
  EXPECT_TRUE(task_seq.PreProcessing());
  EXPECT_TRUE(task_seq.Run());
  EXPECT_TRUE(task_seq.PostProcessing());
  EXPECT_TRUE(CheckTestOutputData(task_seq.GetOutput()));

  // MPI: тестируем только если матрица не пустая
  if (!input_matrix.empty() && !input_matrix[0].empty()) {
    FrolovaSSumElemMatrixMPI task_mpi;
    task_mpi.SetInput(input_matrix);
    EXPECT_TRUE(task_mpi.Validation());
    EXPECT_TRUE(task_mpi.PreProcessing());
    EXPECT_TRUE(task_mpi.Run());
    EXPECT_TRUE(task_mpi.PostProcessing());
    EXPECT_TRUE(CheckTestOutputData(task_mpi.GetOutput()));
  }
}

// Параметры тестов
const std::array<TestType, 6> kTestParam = {std::make_tuple(3, 3, "small"),     std::make_tuple(10, 10, "medium"),
                                            std::make_tuple(20, 15, "rect"),    std::make_tuple(1, 1, "single_element"),
                                            std::make_tuple(200, 200, "large"), std::make_tuple(3, 2, "jagged_matrix")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FrolovaSSumElemMatrixMPI, InType>(kTestParam, PPC_SETTINGS_frolova_s_sum_elem_matrix),
    ppc::util::AddFuncTask<FrolovaSSumElemMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_frolova_s_sum_elem_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = FrolovaSSumElemMatrixRunFuncTests::PrintFuncTestName<FrolovaSSumElemMatrixRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(SumMatrixTests, FrolovaSSumElemMatrixRunFuncTests, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace frolova_s_sum_elem_matrix
