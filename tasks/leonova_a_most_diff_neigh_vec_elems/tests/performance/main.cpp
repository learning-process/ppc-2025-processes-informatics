#include <gtest/gtest.h>

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

    size_t max_diff_position1 = n / 4;
    size_t max_diff_position2 = n / 2;
    size_t max_diff_position3 = 3 * n / 4;

    if (max_diff_position1 + 1 < n) {
      input_data[max_diff_position1] = 1;
      input_data[max_diff_position1 + 1] = 50000;
    }

    if (max_diff_position2 + 1 < n) {
      input_data[max_diff_position2] = 100;
      input_data[max_diff_position2 + 1] = 60000;
      expected_output = std::make_tuple(100, 60000);
    }

    if (max_diff_position3 + 1 < n) {
      input_data[max_diff_position3] = 500;
      input_data[max_diff_position3 + 1] = 45000;
    }

    for (size_t i = 0; i < n / 100; i += 10) {
      if (i + 1 < n) {
        input_data[i] = i % 1000;
        input_data[i + 1] = (i + 500) % 1000;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int diff = std::abs(std::get<0>(output_data) - std::get<1>(output_data));
    return diff >= 50000;
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
