#include <gtest/gtest.h>

#include "util/include/func_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTest : public ppc::util::BaseFunctionalTest<InType, OutType> {
 public:
  InType input_data_{};

  void SetUp() override {
    // формируем тестовую строку
    GlobalData::g_data_string = "Abc123!!!deF#45";

    // input_data_ не влияет, но должен существовать
    input_data_ = 0;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    int cnt = 0;

    for (unsigned char c : GlobalData::g_data_string) {
      if (std::isalpha(c)) {
        cnt++;
      }
    }

    return output_data == cnt;
  }
};

TEST_P(YurkinCountingNumberFuncTest, FunctionalRun) {
  Run();
}

const auto kAllTasks = ppc::util::MakeAllTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSEQ>(
    PPC_SETTINGS_yurkin_counting_number);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllTasks);

const char kFuncTestName[] = "FuncTest";

INSTANTIATE_TEST_SUITE_P(FunctionalTest, YurkinCountingNumberFuncTest, kGtestValues,
                         [](const testing::TestParamInfo<YurkinCountingNumberFuncTest::ParamType> &info) {
                           return std::string(kFuncTestName) + "_" + info.param.name;
                         });

}  // namespace yurkin_counting_number
