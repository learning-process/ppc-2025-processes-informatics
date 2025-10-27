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
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    input_data_ = std::vector<int>(10);
    for(int i = 0;i < input_data_.size();i++)
    {
      input_data_[i] = i;
    }//{0,1,2,3,4,5,6,7,8,9}

    expected_result = std::accumulate(input_data_.begin(),input_data_.end(),0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_result);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result = 0;
};

namespace {

TEST_P(LukinIRunFuncTestsProcesses, ElemVecSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};//idk

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LukinIElemVecSumMPI, InType>(kTestParam, PPC_SETTINGS_lukin_i_elem_vec_sum),
                   ppc::util::AddFuncTask<LukinIElemVecSumSEQ, InType>(kTestParam, PPC_SETTINGS_lukin_i_elem_vec_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LukinIRunFuncTestsProcesses::PrintFuncTestName<LukinIRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ElemVecTests, LukinIRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lukin_i_elem_vec_sum
