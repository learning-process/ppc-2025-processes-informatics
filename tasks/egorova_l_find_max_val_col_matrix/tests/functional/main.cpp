#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"
#include "egorova_l_find_max_val_col_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace egorova_l_find_max_val_col_matrix {

class EgorovaLRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_type = std::get<0>(test_params);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(10, 50);  // Уменьшим диапазон

    switch (test_type) {
      case 0: {  // Пустая матрица
        input_data_ = {};
        break;
      }
      case 1: {  // Большая матрица (4x4)
        input_data_ = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
        break;
      }
      case 2: {  // 1 столбец
        input_data_ = {{1}, {5}, {3}, {7}};
        break;
      }
      case 3: {  // Неквадратная матрица (3x2)
        input_data_ = {{1, 2}, {3, 4}, {5, 6}};
        break;
      }
      case 4: {  // Квадратная матрица (2x2) с известными максимумами
        input_data_ = {{10, 20}, {30, 40}};
        break;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &matrix = GetTestInputData();

    // Отладочный вывод
    std::cout << "Testing matrix: " << matrix.size() << "x" << (matrix.empty() ? 0 : matrix[0].size()) << std::endl;

    if (matrix.empty()) {
      bool result = output_data.empty();
      std::cout << "Empty matrix test: " << (result ? "PASS" : "FAIL") << std::endl;
      return result;
    }

    if (output_data.size() != matrix[0].size()) {
      std::cout << "Size mismatch: expected " << matrix[0].size() << ", got " << output_data.size() << std::endl;
      return false;
    }

    // Вычисляем ожидаемый результат
    std::vector<int> expected(matrix[0].size(), std::numeric_limits<int>::min());
    for (size_t j = 0; j < matrix[0].size(); ++j) {
      for (size_t i = 0; i < matrix.size(); ++i) {
        if (matrix[i][j] > expected[j]) {
          expected[j] = matrix[i][j];
        }
      }
    }

    // Сравниваем результаты
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected[i]) {
        std::cout << "Column " << i << ": expected " << expected[i] << ", got " << output_data[i] << std::endl;
        return false;
      }
    }

    std::cout << "Test PASSED" << std::endl;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(EgorovaLRunFuncTestsProcesses, FindMaxValColMatrix) {
  ExecuteTest(GetParam());
}

// Тестовые случаи: (тип_теста, описание)
const std::array<TestType, 5> kTestParam = {
    std::make_tuple(0, "empty_matrix"), std::make_tuple(1, "large_matrix"), std::make_tuple(2, "single_column"),
    std::make_tuple(3, "non_square_matrix"), std::make_tuple(4, "square_matrix")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<EgorovaLFindMaxValColMatrixMPI, InType>(kTestParam, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<EgorovaLFindMaxValColMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = EgorovaLRunFuncTestsProcesses::PrintFuncTestName<EgorovaLRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixTests, EgorovaLRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace egorova_l_find_max_val_col_matrix
