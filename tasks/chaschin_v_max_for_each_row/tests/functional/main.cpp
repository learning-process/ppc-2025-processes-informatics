#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"
#include "chaschin_v_max_for_each_row/mpi/include/ops_mpi.hpp"
#include "chaschin_v_max_for_each_row/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace chaschin_v_max_for_each_row {

class ChaschinVRunFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    // создаём квадратную матрицу size x size
    TestType params =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);

    input_data_.assign(size, std::vector<float>(size));

    // заполним матрицу
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        input_data_[i][j] = static_cast<float>((i + 1) * (j + 2));
      }
    }

    // заранее считаем эталон — максимум по каждой строке
    expected_output_.resize(size);
    for (int i = 0; i < size; i++) {
      expected_output_[i] = *std::max_element(input_data_[i].begin(), input_data_[i].end());
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) return false;

    for (size_t i = 0; i < output_data.size(); i++) {
      if (output_data[i] != expected_output_[i]) return false;
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

TEST_P(ChaschinVRunFuncTestsProcesses, MaxInRowsFromMatrix) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(3, "3"),
    std::make_tuple(5, "5"),
    std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ChaschinVMaxForEachRow, InType>(kTestParam, PPC_SETTINGS_example_processes),
                   ppc::util::AddFuncTask<ChaschinVMaxForEachRowSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ChaschinVRunFuncTestsProcesses::PrintFuncTestName<ChaschinVRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixRowMaxTests,
                         ChaschinVRunFuncTestsProcesses,
                         kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace chaschin_v_max_for_each_row
