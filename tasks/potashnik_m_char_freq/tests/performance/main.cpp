#include <gtest/gtest.h>

#include "potashnik_m_char_freq/common/include/common.hpp"
#include "potashnik_m_char_freq/mpi/include/ops_mpi.hpp"
#include "potashnik_m_char_freq/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace potashnik_m_char_freq {

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
    ppc::util::MakeAllPerfTasks<InType, PotashnikMCharFreqMPI, PotashnikMCharFreqSEQ>(PPC_SETTINGS_potashnik_m_char_freq);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace potashnik_m_char_freq
