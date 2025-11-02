#include <gtest/gtest.h>

#include "romanova_v_min_by_matrix_rows_processes/common/include/common.hpp"
#include "romanova_v_min_by_matrix_rows_processes/mpi/include/ops_mpi.hpp"
#include "romanova_v_min_by_matrix_rows_processes/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace romanova_v_min_by_matrix_rows_processes {

class RomanovaVMinByMatrixRowsPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data.size() != 0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(RomanovaVMinByMatrixRowsPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RomanovaVMinByMatrixRowsMPI, RomanovaVMinByMatrixRowsSEQ>(PPC_SETTINGS_romanova_v_min_by_matrix_rows_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RomanovaVMinByMatrixRowsPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RomanovaVMinByMatrixRowsPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace romanova_v_min_by_matrix_rows_processes
