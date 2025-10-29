#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "baldin_a_word_count/common/include/common.hpp"
#include "baldin_a_word_count/mpi/include/ops_mpi.hpp"
#include "baldin_a_word_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace baldin_a_word_count {

class BaldinAWordCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data;
  OutType expected_output{};

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_baldin_a_word_count, "book-war-and-peace.txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("File reading error");
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    input_data = buffer.str();
    file.close();

    int count = 0;
    bool in_word = false;
    for (char c : input_data) {
      if (std::isalnum(c) || c == '-' || c == '_') {
        if (!in_word) {
          in_word = true;
          count++;
        }
      } else {
        in_word = false;
      }
    }
    expected_output = count;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_output;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(BaldinAWordCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BaldinAWordCountMPI, BaldinAWordCountSEQ>(PPC_SETTINGS_baldin_a_word_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BaldinAWordCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BaldinAWordCountPerfTests, kGtestValues, kPerfTestName);

}  // namespace baldin_a_word_count
