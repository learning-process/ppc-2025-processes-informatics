#include <gtest/gtest.h>
#include <mpi.h>

#include <string>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "volkov_a_count_word_line/common/include/common.hpp"
#include "volkov_a_count_word_line/mpi/include/ops_mpi.hpp"
#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

namespace volkov_a_count_word_line {

class VolkovACountWordLinePerfTests : public ppc::util::BaseRunPerfTests<std::string, OutType> {
 protected:
  void SetUp() override {
    const int words_count = 500000;
    const std::string word = "word ";

    input_data_.reserve(words_count * word.size());
    for (int i = 0; i < words_count; ++i) {
      input_data_ += word;
    }

    expected_output_ = words_count;
  }

  std::string GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &val) override {
    return val == expected_output_;
  }

 private:
  std::string input_data_;
  OutType expected_output_ = 0;
};

TEST_P(VolkovACountWordLinePerfTests, RunPerformance) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks = ppc::util::MakeAllPerfTasks<std::string, VolkovACountWordLineMPI, VolkovACountWordLineSEQ>(
    PPC_SETTINGS_volkov_a_count_word_line);

const auto kTestParams = ppc::util::TupleToGTestValues(kPerfTasks);
const auto kTestNameGen = VolkovACountWordLinePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(VolkovPerfTests, VolkovACountWordLinePerfTests, kTestParams, kTestNameGen);

}  // namespace volkov_a_count_word_line
