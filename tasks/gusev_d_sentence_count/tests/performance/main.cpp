#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "gusev_d_sentence_count/common/include/common.hpp"
#include "gusev_d_sentence_count/mpi/include/ops_mpi.hpp"
#include "gusev_d_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace gusev_d_sentence_count {

static bool IsTerminator(char c) {
  return c == '.' || c == '!' || c == '?';
}

class GusevDSentenceCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_ = 0;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath("gusev_d_sentence_count", "holy-bible.txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("File reading error");
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    input_data_ = buffer.str();
    file.close();

    size_t count = 0;
    size_t len = input_data_.length();

    for (size_t i = 0; i < len; ++i) {
      if (IsTerminator(input_data_[i])) {
        bool is_next_not_term = (i + 1 == len) || !IsTerminator(input_data_[i + 1]);
        if (is_next_not_term) {
          count++;
        }
      }
    }
    expected_output_ = count;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  GusevDSentenceCountPerfTests() = default;
};

TEST_P(GusevDSentenceCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GusevDSentenceCountMPI, GusevDSentenceCountSEQ>(PPC_SETTINGS_gusev_d_sentence_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GusevDSentenceCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GusevDSentenceCountPerfTests, kGtestValues, kPerfTestName);

}  // namespace gusev_d_sentence_count