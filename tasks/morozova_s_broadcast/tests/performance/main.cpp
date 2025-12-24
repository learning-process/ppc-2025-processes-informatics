#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "morozova_s_broadcast/mpi/include/ops_mpi.hpp"
#include "morozova_s_broadcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozova_s_broadcast {

class MorozovaSRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    auto test_info = GetParam();
    int test_idx = std::get<0>(test_info);
    const int sizes[] = {100, 1000, 10000, 100000, 1000000};
    int size = sizes[test_idx % 5];
    input_data_.resize(size);
    std::iota(input_data_.begin(), input_data_.end(), 0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == input_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MorozovaSRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MorozovaSBroadcastMPI, MorozovaSBroadcastSEQ>(
    PPC_SETTINGS_morozova_s_broadcast);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = MorozovaSRunPerfTestProcesses::CustomPerfTestName;
INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovaSRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace morozova_s_broadcast
