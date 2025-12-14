#include <gtest/gtest.h>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "levonychev_i_multistep_2d_optimization/mpi/include/ops_mpi.hpp"
#include "levonychev_i_multistep_2d_optimization/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace levonychev_i_multistep_2d_optimization {

class LevonychevIMultistep2dOptimizationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(LevonychevIMultistep2dOptimizationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LevonychevIMultistep2dOptimizationMPI, LevonychevIMultistep2dOptimizationSEQ>(PPC_SETTINGS_levonychev_i_multistep_2d_optimization);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LevonychevIMultistep2dOptimizationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LevonychevIMultistep2dOptimizationPerfTests, kGtestValues, kPerfTestName);

}  // namespace levonychev_i_multistep_2d_optimization
