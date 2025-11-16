#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "nikitina_v_max_elem_matr/common/include/common.hpp"
#include "nikitina_v_max_elem_matr/mpi/include/ops_mpi.hpp"
#include "nikitina_v_max_elem_matr/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nikitina_v_max_elem_matr {

using TestType = std::vector<int>;

class NikitinaVMaxElemMatrFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<ParamType> &info) {
    auto task_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
    std::string tech = (task_name.find("seq") != std::string::npos) ? "seq" : "mpi";
    return tech + "_" + std::to_string(info.index);
  }

 protected:
  void SetUp() override {
    input_data_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    bool is_valid = true;
    if (input_data_.size() < 2) {
      is_valid = false;
    } else {
      int rows = input_data_[0];
      int cols = input_data_[1];
      if (rows < 0 || cols < 0 || static_cast<size_t>(rows * cols) != input_data_.size() - 2) {
        is_valid = false;
      }
    }

    if (is_valid) {
      if (input_data_.size() == 2) {
        expected_output_ = INT_MIN;
      } else {
        expected_output_ = *std::max_element(input_data_.begin() + 2, input_data_.end());
      }
    } else {
      expected_output_ = 0;
    }
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return (output_data == expected_output_);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_output_{};
};

namespace {

// ============================================================================
// ==================== ИСПРАВЛЕННАЯ ЛОГИКА ТЕСТА ============================
// ============================================================================
TEST_P(NikitinaVMaxElemMatrFuncTests, FindMaxElement) {
  auto test_param = GetParam();
  auto task_getter = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTaskGetter)>(test_param);
  const auto &input_data = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(test_param);

  auto task = task_getter(input_data);

  bool validation_should_fail = false;
  if (input_data.size() < 2) {
    validation_should_fail = true;
  } else {
    int rows = input_data[0];
    int cols = input_data[1];
    if (rows < 0 || cols < 0 || static_cast<size_t>(rows * cols) != input_data.size() - 2) {
      validation_should_fail = true;
    }
  }

  if (validation_should_fail) {
    ASSERT_FALSE(task->Validation());
    // Все равно вызываем остальные этапы, чтобы деструктор был "доволен"
    task->PreProcessing();
    task->Run();
    task->PostProcessing();
  } else {
    ASSERT_TRUE(task->Validation());
    ASSERT_TRUE(task->PreProcessing());
    ASSERT_TRUE(task->Run());
    ASSERT_TRUE(task->PostProcessing());
    ASSERT_TRUE(CheckTestOutputData(task->GetOutput()));
  }
}

InType generate_matrix(int rows, int cols, int seed = 1) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> distrib(-1000, 1000);
  InType matrix(2 + rows * cols);
  matrix[0] = rows;
  matrix[1] = cols;
  for (int i = 0; i < rows * cols; ++i) {
    matrix[i + 2] = distrib(gen);
  }
  return matrix;
}

const std::array<TestType, 14> kTestCases = {{// --- Валидные случаи ---
                                              generate_matrix(10, 10),
                                              generate_matrix(5, 7),
                                              generate_matrix(1, 1),
                                              generate_matrix(10, 0),
                                              generate_matrix(0, 10),
                                              generate_matrix(0, 0),

                                              // --- Детерминированные случаи для покрытия веток в цикле ---
                                              {3, 2, 100, 1, 2, 3, 4, 5},
                                              {3, 2, 1, 2, 3, 4, 5, 100},
                                              {3, 2, 5, 5, 5, 5, 5, 5},

                                              // --- Невалидные случаи ---
                                              {1},
                                              {-1, 5},
                                              {5, -1},
                                              {2, 2, 1, 2, 3},
                                              {2, 2, 1, 2, 3, 4, 5}}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MaxElementMatrSEQ, InType>(kTestCases, PPC_SETTINGS_nikitina_v_max_elem_matr),
    ppc::util::AddFuncTask<MaxElementMatrMPI, InType>(kTestCases, PPC_SETTINGS_nikitina_v_max_elem_matr));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(NikitinaV_MaxElementMatr_Func, NikitinaVMaxElemMatrFuncTests, kGtestValues,
                         NikitinaVMaxElemMatrFuncTests::PrintTestParam);

}  // namespace

}  // namespace nikitina_v_max_elem_matr
