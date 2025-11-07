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

#include "lukin_i_elem_vec_sum/common/include/common.hpp"
#include "lukin_i_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "lukin_i_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lukin_i_elem_vec_sum {

class LukinIRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string test_name = std::get<1>(test_param);
    return test_name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    size_t vec_size = static_cast<size_t>(std::get<0>(params));
    input_data_ = std::vector<int>(vec_size, 1);

    expected = vec_size;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected = 0;
};

namespace {

TEST_P(LukinIRunFuncTestsProcesses, ElemVecSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 32> kTestParam = {
    std::make_tuple(0, "empty"),         std::make_tuple(1, "single"),         std::make_tuple(2, "pair"),
    std::make_tuple(3, "triple"),        std::make_tuple(4, "quartet"),        std::make_tuple(5, "small_odd"),
    std::make_tuple(6, "small_even"),    std::make_tuple(7, "prime_small"),    std::make_tuple(8, "octet"),
    std::make_tuple(9, "square_small"),  std::make_tuple(10, "medium"),        std::make_tuple(11, "prime_medium"),
    std::make_tuple(12, "dozen"),        std::make_tuple(13, "prime_baker"),   std::make_tuple(14, "even_medium"),
    std::make_tuple(15, "odd_medium"),   std::make_tuple(16, "hex"),           std::make_tuple(17, "prime_hex"),
    std::make_tuple(18, "even_large"),   std::make_tuple(19, "prime_large"),   std::make_tuple(20, "score"),
    std::make_tuple(21, "odd_large"),    std::make_tuple(22, "even_big"),      std::make_tuple(23, "prime_big"),
    std::make_tuple(24, "day_hours"),    std::make_tuple(25, "square_medium"), std::make_tuple(30, "month_days"),
    std::make_tuple(32, "double_hex"),   std::make_tuple(36, "square_large"),  std::make_tuple(40, "forty"),
    std::make_tuple(50, "half_hundred"), std::make_tuple(64, "double_square")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LukinIElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_lukin_i_elem_vec_sum),
                   ppc::util::AddFuncTask<LukinIElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_lukin_i_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LukinIRunFuncTestsProcesses::PrintFuncTestName<LukinIRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ElemVecTests, LukinIRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lukin_i_elem_vec_sum
