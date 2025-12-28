#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "chaschin_v_sobel_operator/common/include/common.hpp"
#include "chaschin_v_sobel_operator/mpi/include/ops_mpi.hpp"
#include "chaschin_v_sobel_operator/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chaschin_v_sobel_operator {

class ChaschinVRunFuncTestsProcessesSO : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    // Генерация входных данных
    input_data_.resize(size);
    for (int i = 0; i < size; ++i) {
      input_data_[i].resize(size);
    }

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        // Детерминированное значение
        input_data_[i][j] = static_cast<float>((i + 1) * (j + 2));
      }
    }

    // Генерация ожидаемого результата
    expected_output_.resize(size);
    for (int i = 0; i < size; i++) {
      expected_output_[i] = *std::max_element(input_data_[i].begin(), input_data_[i].end());
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); i++) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(ChaschinVRunFuncTestsProcessesSO, SobelOperator) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 20> kTestParam = {
    std::make_tuple(4, "4"),   std::make_tuple(5, "5"),   std::make_tuple(7, "7"),   std::make_tuple(8, "8"),
    std::make_tuple(9, "9"),   std::make_tuple(10, "10"), std::make_tuple(11, "11"), std::make_tuple(13, "13"),
    std::make_tuple(15, "15"), std::make_tuple(17, "17"), std::make_tuple(19, "19"), std::make_tuple(21, "21"),
    std::make_tuple(23, "23"), std::make_tuple(25, "25"), std::make_tuple(27, "27"), std::make_tuple(29, "29"),
    std::make_tuple(31, "31"), std::make_tuple(33, "33"), std::make_tuple(35, "35"), std::make_tuple(37, "37")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ChaschinVSobelOperatorMPI, InType>(kTestParam, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<ChaschinVSobelOperatorSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ChaschinVRunFuncTestsProcessesSO::PrintFuncTestName<ChaschinVRunFuncTestsProcessesSO>;

INSTANTIATE_TEST_SUITE_P(MatrixRowMaxTests, ChaschinVRunFuncTestsProcessesSO, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace chaschin_v_sobel_operator
