#include <gtest/gtest.h>

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

using TestType = std::tuple<int, int, int>;

class NikitinaVMaxElemMatrFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  // ====================== КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ======================
  static std::string PrintTestParam(const testing::TestParamInfo<ParamType> &info) {
    // Получаем имя таска (например, "nikitina_v_max_elem_matr_seq_enabled")
    auto task_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);

    // Извлекаем технологию (seq или mpi) из имени
    std::string tech = "unknown";
    if (task_name.find("seq") != std::string::npos) {
      tech = "seq";
    }
    if (task_name.find("mpi") != std::string::npos) {
      tech = "mpi";
    }

    // Добавляем технологию в имя теста, чтобы оно стало уникальным
    std::string test_name = "tech_" + tech + "_test_id_" + std::to_string(std::get<0>(params)) + "_rows_" +
                            std::to_string(std::get<1>(params)) + "_cols_" + std::to_string(std::get<2>(params));
    return test_name;
  }
  // =================================================================

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int rows = std::get<1>(params);
    int cols = std::get<2>(params);

    if (rows <= 0 || cols <= 0) {
      input_data_ = {rows, cols};
      expected_output_ = INT_MIN;
      return;
    }

    std::mt19937 gen(1);
    std::uniform_int_distribution<> distrib(-1000, 1000);
    int max_val = INT_MIN;

    InType generated_matr(2 + rows * cols);
    generated_matr[0] = rows;
    generated_matr[1] = cols;

    for (int i = 0; i < rows * cols; ++i) {
      generated_matr[i + 2] = distrib(gen);
      if (i == 0 || generated_matr[i + 2] > max_val) {
        max_val = generated_matr[i + 2];
      }
    }

    input_data_ = generated_matr;
    expected_output_ = max_val;
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

TEST_P(NikitinaVMaxElemMatrFuncTests, FindMaxElement) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(1, 10, 10), std::make_tuple(2, 5, 15),
                                            std::make_tuple(3, 1, 30), std::make_tuple(4, 30, 1),
                                            std::make_tuple(5, 1, 1)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MaxElementMatrSEQ, InType>(kTestParam, PPC_SETTINGS_nikitina_v_max_elem_matr),
    ppc::util::AddFuncTask<MaxElementMatrMPI, InType>(kTestParam, PPC_SETTINGS_nikitina_v_max_elem_matr));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(NikitinaV_MaxElementMatr_Func, NikitinaVMaxElemMatrFuncTests, kGtestValues,
                         NikitinaVMaxElemMatrFuncTests::PrintTestParam);

}  // namespace
}  // namespace nikitina_v_max_elem_matr
