#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"
#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"
#include "yurkin_counting_number/seq/include/ops_seq.hpp"

namespace yurkin_counting_number {
void SetTextForInput(const InType key, const std::string &text);
const std::string &GetTextForInput(const InType key);
}  // namespace yurkin_counting_number

namespace yurkin_counting_number {

class YurkinCountingNumberPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};
  void SetUp() override {
    input_data_ = kCount_;

    std::string text;
    text.reserve(static_cast<size_t>(input_data_) + 16);
    for (int i = 0; i < input_data_; ++i) {
      text.push_back(char('A' + (i % 26)));

      if ((i % 10) == 9) {
        text.push_back('1');
      }
    }

    text += " 123!";

    int letters = 0;
    for (unsigned char c : text) {
      if (std::isalpha(c)) {
        ++letters;
      }
    }

    if (letters > input_data_) {
      while (letters > input_data_ && !text.empty()) {
        char ch = text.back();
        text.pop_back();
        if (std::isalpha(static_cast<unsigned char>(ch))) {
          --letters;
        }
      }
    } else if (letters < input_data_) {
      while (letters < input_data_) {
        text.push_back('B');
        ++letters;
      }
    }

    SetTextForInput(input_data_, text);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }
  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(YurkinCountingNumberPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YurkinCountingNumberMPI, YurkinCountingNumberSEQ>(
    PPC_SETTINGS_yurkin_counting_number);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = YurkinCountingNumberPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YurkinCountingNumberPerfTest, kGtestValues, kPerfTestName);

}  // namespace yurkin_counting_number
