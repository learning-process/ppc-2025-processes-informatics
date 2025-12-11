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
void SetTextForInput(const InType key, const std::string &text);
const std::string &GetTextForInput(const InType key);
}  // namespace yurkin_counting_number

namespace yurkin_counting_number {

class YurkinCountingNumberFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int base = std::get<0>(params);

    std::string text;
    text.reserve(static_cast<size_t>(base) + 8);
    for (int i = 0; i < base; ++i) {
      text.push_back('A');
    }
    text += " 123!";

    int letters = 0;
    for (unsigned char c : text) {
      if (std::isalpha(c)) {
        ++letters;
      }
    }
    if (letters != base) {
      if (letters > base) {
        while (letters > base && !text.empty()) {
          char ch = text.back();
          text.pop_back();
          if (std::isalpha(static_cast<unsigned char>(ch))) {
            --letters;
          }
        }
      } else {
        while (letters < base) {
          text.push_back('C');
          ++letters;
        }
      }
    }

    SetTextForInput(base, text);

    input_data_ = base;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
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
