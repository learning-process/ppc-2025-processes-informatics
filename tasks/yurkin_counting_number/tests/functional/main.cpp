#include <gtest/gtest.h>

#include <cctype>

#include "util/include/func_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);

    GlobalData::g_data_string.clear();
    GlobalData::g_data_string.resize(2'000'000, 'a');

    for (int i = 0; i < input_data_; i++) {
      GlobalData::g_data_string[i] = 'X';
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(YurkinCountingNumberFuncTests, RunFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinCountingNumberMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number),
    ppc::util::AddFuncTask<YurkinCountingNumberSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = YurkinCountingNumberFuncTests::PrintFuncTestName<YurkinCountingNumberFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, YurkinCountingNumberFuncTests, kGtestValues, kPerfTestName);

}  // namespace yurkin_counting_number
