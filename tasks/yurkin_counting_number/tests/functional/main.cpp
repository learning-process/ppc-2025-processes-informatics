#include <gtest/gtest.h>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);

    GlobalData::g_data_string.clear();
    GlobalData::g_data_string = input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int expected = 0;
    for (char c : input_data_) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        expected++;
      }
    }
    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

const std::array<TestType, 3> kTestParam = {std::make_tuple(0, std::string("AbC123")),
                                            std::make_tuple(1, std::string("!!!!abcd")),
                                            std::make_tuple(2, std::string("123"))};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinCountingNumberMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number),
    ppc::util::AddFuncTask<YurkinCountingNumberSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number));

const auto kValues = ppc::util::ExpandToValues(kTasks);

const auto kName = YurkinCountingNumberFuncTests::PrintFuncTestName<YurkinCountingNumberFuncTests>;

TEST_P(YurkinCountingNumberFuncTests, MainTest) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(YurkinTests, YurkinCountingNumberFuncTests, kValues, kName);

}  // namespace

}  // namespace yurkin_counting_number
