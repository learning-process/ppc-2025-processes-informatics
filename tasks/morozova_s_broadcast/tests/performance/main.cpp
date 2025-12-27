#include <gtest/gtest.h>

#include <array>
#include <cstddef>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "morozova_s_broadcast/mpi/include/ops_mpi.hpp"
#include "morozova_s_broadcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozova_s_broadcast {

class MorozovaSRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    static int test_counter = 0;
    constexpr std::array<int, 5> kSizes = {100, 1000, 10000, 100000, 1000000};
    const int test_idx = test_counter++ % 5;
    const int size = kSizes[static_cast<std::size_t>(test_idx)];

    input_data_.resize(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) {
      input_data_[static_cast<std::size_t>(i)] = i;
    }
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
