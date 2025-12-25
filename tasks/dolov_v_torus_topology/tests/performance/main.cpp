#include <gtest/gtest.h>

#include "dolov_v_torus_topology/common/include/common.hpp"
#include "dolov_v_torus_topology/mpi/include/ops_mpi.hpp"
#include "dolov_v_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dolov_v_torus_topology {

class DolovVTorusTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(DolovVTorusTopologyPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DolovVTorusTopologyMPI, DolovVTorusTopologySEQ>(
    PPC_SETTINGS_dolov_v_torus_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DolovVTorusTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DolovVTorusTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace dolov_v_torus_topology
