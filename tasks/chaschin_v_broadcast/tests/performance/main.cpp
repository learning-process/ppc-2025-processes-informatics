#include <gtest/gtest.h>

#include "example_processes/common/include/common.hpp"
#include "example_processes/mpi/include/ops_mpi.hpp"
#include "example_processes/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chaschin_v_broadcast {

class ChaschinVRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000000;
  InType input_data_{};

  void SetUp() override {
    input_data_.resize(kCount_, 7863453);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::equal(input_data_.begin(), input_data_.end(), output_data.begin());
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChaschinVRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChaschinVBroadcastMPI, ChaschinVBroadcastSEQ>(PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChaschinVRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChaschinVRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace chaschin_v_broadcast
