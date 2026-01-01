#include <gtest/gtest.h>

#include <vector>

#include "agafonov_i_torus_grid/common/include/common.hpp"
#include "agafonov_i_torus_grid/mpi/include/ops_mpi.hpp"
#include "agafonov_i_torus_grid/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace agafonov_i_torus_grid {

class TorusGridPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = {12345, 0, 3};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 12345;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(TorusGridPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TorusGridTaskMPI, TorusGridTaskSEQ>(PPC_SETTINGS_agafonov_i_torus_grid);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TorusGridPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(TorusGridPerfTests, TorusGridPerfTests, kGtestValues, kPerfTestName);

}  // namespace agafonov_i_torus_grid
