#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "util/include/perf_test_util.hpp"
#include "volkov_a_count_word_line/common/include/common.hpp"
#include "volkov_a_count_word_line/mpi/include/ops_mpi.hpp"
#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

namespace volkov_a_count_word_line {

class VolkovACountWordLinePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int words_count = 25000000;
    const std::string word = "word ";

    input_data_.resize(words_count * word.size());
    size_t pattern_len = word.size();
    for (size_t i = 0; i < input_data_.size(); ++i) {
      input_data_[i] = word[i % pattern_len];
    }

    expected_output_ = words_count;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &val) override {
    return val == expected_output_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

TEST_P(VolkovACountWordLinePerfTests, RunPerformance) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<InType, VolkovACountWordLineMPI, VolkovACountWordLineSEQ>(
    PPC_SETTINGS_volkov_a_count_word_line);

const auto kTestParams = ppc::util::TupleToGTestValues(kPerfTasks);
const auto kTestNameGen = VolkovACountWordLinePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(VolkovPerfTests, VolkovACountWordLinePerfTests, kTestParams, kTestNameGen);

}  // namespace volkov_a_count_word_line
