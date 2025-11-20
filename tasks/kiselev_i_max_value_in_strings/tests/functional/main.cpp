#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"
#include "kiselev_i_max_value_in_strings/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_max_value_in_strings {

class KiselevIRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType&) {
    static int test_counter = 0;
    return "test_" + std::to_string(test_counter++);
}


 protected:

  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_      = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(KiselevIRunFuncTestsProcesses, MaxInRowsFromMatrix) {
  ExecuteTest(GetParam());
}

// Тестовые случаи: (input_matrix, expected_max_values)
const std::array<TestType, 4> kTestParam = {
  // Тест 1: Простая матрица 3x3
  std::make_tuple(
    std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
    std::vector<int>{3, 6, 9}
  ),
  
  // Тест 2: С отрицательными числами
  std::make_tuple(
    std::vector<std::vector<int>>{{-10, -50, -30}, {10, 20, 35}, {0, 0, 0}},
    std::vector<int>{-10, 35, 0}
  ),
  
  // Тест 3: Один элемент
  std::make_tuple(
    std::vector<std::vector<int>>{{228}},
    std::vector<int>{228}
  ),
  
  // Тест 4: Рваная матрица
  std::make_tuple(
    std::vector<std::vector<int>>{{1}, {2 ,3, 4, 5, 6}, {7, 8}},
    std::vector<int>{1, 6, 8}
  )
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KiselevITestTaskMPI, InType>(kTestParam, PPC_SETTINGS_kiselev_i_max_value_in_strings),
                   ppc::util::AddFuncTask<KiselevITestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_kiselev_i_max_value_in_strings));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KiselevIRunFuncTestsProcesses::PrintFuncTestName<KiselevIRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixRowMaxTests, KiselevIRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kiselev_i_max_value_in_strings

