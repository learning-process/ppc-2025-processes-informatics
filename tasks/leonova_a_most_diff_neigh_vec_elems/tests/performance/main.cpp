#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "leonova_a_most_diff_neigh_vec_elems/common/include/common.hpp"
#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"
#include "leonova_a_most_diff_neigh_vec_elems/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace leonova_a_most_diff_neigh_vec_elems {

class MostDiffNeighVecElemsRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t n = 30000000;
  InType input_data{};
  OutType expected_output{};

  void SetUp() override {
    input_data.resize(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-100000, 100000);

    for (size_t i = 0; i < n; ++i) {
      input_data[i] = dist(gen);
    }

    // СЛУЧАЙНАЯ поз для макс
    std::uniform_int_distribution<size_t> pos_dist(0, n - 2);  // От 0 до n-2 (чтобы была пара)
    size_t max_diff_position = pos_dist(gen);

    // гарантированно макс пара
    input_data[max_diff_position] = -150000;
    input_data[max_diff_position + 1] = 150000;
    expected_output = std::make_tuple(-150000, 150000);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output == output_data);
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(MostDiffNeighVecElemsRunPerfTestsProcesses, RunPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LeonovaAMostDiffNeighVecElemsMPI, LeonovaAMostDiffNeighVecElemsSEQ>(
        PPC_SETTINGS_leonova_a_most_diff_neigh_vec_elems);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MostDiffNeighVecElemsRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunVecPerf, MostDiffNeighVecElemsRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace leonova_a_most_diff_neigh_vec_elems
