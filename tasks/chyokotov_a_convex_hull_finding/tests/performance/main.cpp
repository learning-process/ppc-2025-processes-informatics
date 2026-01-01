#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"
#include "chyokotov_a_convex_hull_finding/mpi/include/ops_mpi.hpp"
#include "chyokotov_a_convex_hull_finding/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chyokotov_a_convex_hull_finding {

class ChyokotovConvexHullPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_;

  void SetUp() override {}

  bool CheckTestOutputData(OutType &output_data) final {
    size_t points_actual = 0;
    size_t points_expected = 0;

    for (const auto &i : output_data) {
      points_actual += i.size();
    }

    for (const auto &i : expected_) {
      points_expected += i.size();
    }

    if (points_actual != points_expected) {
      return false;
    }

    std::multiset<std::pair<int, int>> set_actual;
    std::multiset<std::pair<int, int>> set_expected;

    for (const auto &i : output_data) {
      set_actual.insert(i.begin(), i.end());
    }

    for (const auto &i : expected_) {
      set_expected.insert(i.begin(), i.end());
    }

    if (set_actual != set_expected) {
      return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChyokotovConvexHullPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChyokotovConvexHullFindingMPI, ChyokotovConvexHullFindingSEQ>(
        PPC_SETTINGS_chyokotov_a_convex_hull_finding);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChyokotovConvexHullPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChyokotovConvexHullPerfTest, kGtestValues, kPerfTestName);

}  // namespace chyokotov_a_convex_hull_finding
