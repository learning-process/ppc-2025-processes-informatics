#include <gtest/gtest.h>

#include <array>
#include <string>
#include <tuple>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    TestType test = std::get<2>(GetParam());
    const std::string &s = std::get<1>(test);
    input_data_.assign(s.begin(), s.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected = 0;
    for (char c : input_data_) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        expected++;
      }
    }
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(YurkinCountingNumberFuncTest, CountLetters) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "abc123"), std::make_tuple(5, "a1b2c"),
                                            std::make_tuple(7, "ABCDEFG")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinCountingNumberMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number),
    ppc::util::AddFuncTask<YurkinCountingNumberSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = YurkinCountingNumberFuncTest::PrintFuncTestName<YurkinCountingNumberFuncTest>;

INSTANTIATE_TEST_SUITE_P(CountLettersTests, YurkinCountingNumberFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yurkin_counting_number
