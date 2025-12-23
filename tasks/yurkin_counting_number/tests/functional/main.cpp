#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string name = std::get<1>(test_param);
    for (auto &c : name) {
      if (std::isalnum(static_cast<unsigned char>(c)) == 0) {
        c = '_';
      }
    }
    return name;
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<1>(GetParam());
    input_.assign(params.begin(), params.end());
  }

  bool CheckTestOutputData(OutType &output) final {
    int expected = 0;
    for (unsigned char c : input_) {
      if (std::isalpha(c) != 0) {
        ++expected;
      }
    }
    return expected == output;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
};

namespace {

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(0, ""),           std::make_tuple(1, "a"),
    std::make_tuple(2, "A1!"),        std::make_tuple(3, "abcXYZ"),
    std::make_tuple(4, "123456"),     std::make_tuple(5, "   \n\t"),
    std::make_tuple(6, "!@#$%^&*()"), std::make_tuple(7, std::string(1000, 'a'))};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinCountingNumberMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number),
    ppc::util::AddFuncTask<YurkinCountingNumberSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number));

const auto kValues = ppc::util::ExpandToValues(kTasks);

const auto kName = [](const ::testing::TestParamInfo<typename YurkinCountingNumberFuncTests::ParamType> &info) {
  return YurkinCountingNumberFuncTests::PrintTestParam(std::get<1>(info.param));
};

TEST_P(YurkinCountingNumberFuncTests, MainTest) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(YurkinTests, YurkinCountingNumberFuncTests, kValues, kName);

}  // namespace
}  // namespace yurkin_counting_number
