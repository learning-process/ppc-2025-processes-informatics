#include <gtest/gtest.h>

#include "frolova_s_star_topology/common/include/common.hpp"
#include "frolova_s_star_topology/mpi/include/ops_mpi.hpp"
#include "frolova_s_star_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace frolova_s_star_topology {

class FrolovaSRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 2147483645;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(FrolovaSRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FrolovaSStarTopologyMPI, FrolovaSStarTopologySEQ>(
    PPC_SETTINGS_frolova_s_star_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FrolovaSRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FrolovaSRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace frolova_s_star_topology
