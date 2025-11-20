#include <gtest/gtest.h>

#include "kichanova_k_count_letters_in_str/common/include/common.hpp"
#include "kichanova_k_count_letters_in_str/mpi/include/ops_mpi.hpp"
#include "kichanova_k_count_letters_in_str/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kichanova_k_count_letters_in_str {

class KichanovaKCountLettersInStrPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const std::string kTestString_ = "test string with letters and numbers 12345 and symbols !@#$%";
  InType input_data_{};
  int expected_output_{};

  void SetUp() override {
    input_data_ = kTestString_;
    expected_output_ = 0;
    for (char c : kTestString_) {
      if (std::isalpha(static_cast<unsigned char>(c))) {
        expected_output_++;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KichanovaKCountLettersInStrPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KichanovaKCountLettersInStrMPI, KichanovaKCountLettersInStrSEQ>(PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KichanovaKCountLettersInStrPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KichanovaKCountLettersInStrPerfTest, kGtestValues, kPerfTestName);


}  // namespace kichanova_k_count_letters_in_str
