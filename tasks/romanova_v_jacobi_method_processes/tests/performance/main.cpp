#include <gtest/gtest.h>

#include "romanova_v_jacobi_method_processes/common/include/common.hpp"
#include "romanova_v_jacobi_method_processes/mpi/include/ops_mpi.hpp"
#include "romanova_v_jacobi_method_processes/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanova_v_jacobi_method_processes {

class RomanovaVJacobiMethodPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RomanovaVJacobiMethodPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RomanovaVJacobiMethodMPI, RomanovaVJacobiMethodSEQ>(PPC_SETTINGS_romanova_v_jacobi_method_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RomanovaVJacobiMethodPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovaVJacobiMethodPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace romanova_v_jacobi_method_processes
