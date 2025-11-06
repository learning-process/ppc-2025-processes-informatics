#include <gtest/gtest.h>

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

#include "smyshlaev_a_str_order_check/common/include/common.hpp"
#include "smyshlaev_a_str_order_check/mpi/include/ops_mpi.hpp"
#include "smyshlaev_a_str_order_check/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_str_order_check {

class SmyshlaevAStrOrderCheckRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &input = std::get<0>(test_param);
    const auto &expected = std::get<1>(test_param);
    return input.first + "_" + input.second + "_expect_" + std::to_string(expected);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final { return (expected_output_ == output_data); }

  InType GetTestInputData() final { return input_data_; }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(SmyshlaevAStrOrderCheckRunFuncTestsProcesses, StringOrderCheckTest) { ExecuteTest(GetParam()); }

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(std::make_pair("apple", "apple"), 0),
    std::make_tuple(std::make_pair("apple", "banana"), -1),
    std::make_tuple(std::make_pair("zebra", "yak"), 1),
    std::make_tuple(std::make_pair("cat", "caterpillar"), -1),
    std::make_tuple(std::make_pair("caterpillar", "cat"), 1),
    std::make_tuple(std::make_pair("Zebra", "zebra"), -1)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<SmyshlaevAStrOrderCheckMPI, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_str_order_check),
                   ppc::util::AddFuncTask<SmyshlaevAStrOrderCheckSEQ, InType>(kTestParam, PPC_SETTINGS_smyshlaev_a_str_order_check));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SmyshlaevAStrOrderCheckRunFuncTestsProcesses::PrintFuncTestName<SmyshlaevAStrOrderCheckRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(StringOrderCheckTests, SmyshlaevAStrOrderCheckRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace smyshlaev_a_str_order_check