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

    int vec_size = static_cast<int>(std::get<0>(params));
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

const std::array<TestType, 29> kTestParam = {std::make_tuple(0, "empty_vector"),
                                             std::make_tuple(1, "single_element"),
                                             std::make_tuple(2, "two_elements"),
                                             std::make_tuple(3, "small_odd"),
                                             std::make_tuple(4, "small_even"),
                                             std::make_tuple(7, "prime_small"),
                                             std::make_tuple(8, "power_of_two_small"),
                                             std::make_tuple(15, "medium_odd"),
                                             std::make_tuple(16, "power_of_two_medium"),
                                             std::make_tuple(31, "prime_medium"),
                                             std::make_tuple(32, "power_of_two_large"),
                                             std::make_tuple(50, "round_number_medium"),
                                             std::make_tuple(63, "odd_boundary"),
                                             std::make_tuple(64, "power_of_two_boundary"),
                                             std::make_tuple(100, "round_number_large"),
                                             std::make_tuple(127, "prime_large"),
                                             std::make_tuple(255, "odd_large"),
                                             std::make_tuple(256, "power_of_two_xlarge"),
                                             std::make_tuple(1000, "large_round"),
                                             std::make_tuple(1023, "very_large_odd"),
                                             std::make_tuple(10000, "extreme_size"),
                                             std::make_tuple(16384, "large_power_of_two"),
                                             std::make_tuple(32767, "max_typical_odd"),
                                             std::make_tuple(50000, "large_arbitrary"),
                                             std::make_tuple(99999, "large_odd_arbitrary"),
                                             std::make_tuple(100000, "very_large_round"),
                                             std::make_tuple(131072, "huge_power_of_two"),
                                             std::make_tuple(500000, "huge_arbitrary"),
                                             std::make_tuple(1000000, "million_elements")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LukinIElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_lukin_i_elem_vec_sum),
                   ppc::util::AddFuncTask<LukinIElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_lukin_i_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LukinIRunFuncTestsProcesses::PrintFuncTestName<LukinIRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ElemVecTests, LukinIRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lukin_i_elem_vec_sum
