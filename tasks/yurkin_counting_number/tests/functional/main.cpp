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

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const auto &param = GetParam();
    TestType test = std::get<2>(param);

    const std::string &s = std::get<1>(test);
    input_data_.assign(s.begin(), s.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {
TEST_P(YurkinCountingNumberFuncTest, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<YurkinCountingNumberMPI, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number),
    ppc::util::AddFuncTask<YurkinCountingNumberSEQ, InType>(kTestParam, PPC_SETTINGS_yurkin_counting_number));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = YurkinCountingNumberFuncTest::PrintFuncTestName<YurkinCountingNumberFuncTest>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, YurkinCountingNumberFuncTest, kGtestValues, kPerfTestName);
}  // namespace

}  // namespace yurkin_counting_number
