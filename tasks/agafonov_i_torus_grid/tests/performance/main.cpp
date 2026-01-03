#include <gtest/gtest.h>

#include "agafonov_i_torus_grid/common/include/common.hpp"
#include "agafonov_i_torus_grid/mpi/include/ops_mpi.hpp"
#include "agafonov_i_torus_grid/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace agafonov_i_torus_grid {

class TorusGridPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  TorusGridPerfTests() = default;

 protected:
  void SetUp() override {
    input_data_ = {.value = 12345, .source_rank = 0, .dest_rank = 0};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 12345;
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{.value = 0, .source_rank = 0, .dest_rank = 0};
};

TEST_P(TorusGridPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TorusGridTaskMPI, TorusGridTaskSEQ>(PPC_SETTINGS_agafonov_i_torus_grid);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = TorusGridPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(TorusGridPerfTests, TorusGridPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace agafonov_i_torus_grid
