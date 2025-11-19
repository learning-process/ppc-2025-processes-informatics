#include <gtest/gtest.h>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"
#include "kruglova_a_max_diff_adjacent/mpi/include/ops_mpi.hpp"
#include "kruglova_a_max_diff_adjacent/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kruglova_a_max_diff_adjacent {

class KruglovaAMaxDiffAdjacentPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(KruglovaAMaxDiffAdjacentPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KruglovaAMaxDiffAdjacentMPI, KruglovaAMaxDiffAdjacentSEQ>(PPC_SETTINGS_kruglova_a_max_diff_adjacent);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KruglovaAMaxDiffAdjacentPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KruglovaAMaxDiffAdjacentPerfTests, kGtestValues, kPerfTestName);

}  // namespace kruglova_a_max_diff_adjacent
