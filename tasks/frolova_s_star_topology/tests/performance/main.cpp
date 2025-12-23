#include <gtest/gtest.h>

#include <array>
#include <tuple>

#include "frolova_s_star_topology/common/include/common.hpp"
#include "frolova_s_star_topology/mpi/include/ops_mpi.hpp"
#include "util/include/perf_test_util.hpp"

namespace frolova_s_star_topology {

class FrolovaSStarTopologyPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // Для performance тестов используем фиксированный destination
    input_data_ = 1;  // destination = 1
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(FrolovaSStarTopologyPerfTests, RunPerformanceTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, FrolovaSStarTopologyMPI>(PPC_SETTINGS_frolova_s_star_topology);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = FrolovaSStarTopologyPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(FrolovaSStarTopologyPerf, FrolovaSStarTopologyPerfTests, kGtestValues, kPerfTestName);

}  // namespace frolova_s_star_topology
