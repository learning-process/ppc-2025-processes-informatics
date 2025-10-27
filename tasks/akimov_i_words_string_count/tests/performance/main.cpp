#include <gtest/gtest.h>

#include "akimov_i_words_string_count/common/include/common.hpp"
#include "akimov_i_words_string_count/mpi/include/ops_mpi.hpp"
#include "akimov_i_words_string_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace akimov_i_words_string_count {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, AkimovIWordsStringCountMPI, AkimovIWordsStringCountSEQ>(PPC_SETTINGS_akimov_i_words_string_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace akimov_i_words_string_count
