#include <gtest/gtest.h>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/mpi/include/ops_mpi.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

class NikolaevDMostDifVecNeighborsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(NikolaevDMostDifVecNeighborsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikolaevDMostDifVecNeighborsMPI, NikolaevDMostDifVecNeighborsSEQ>(PPC_SETTINGS_nikolaev_d_most_dif_vec_neighbors);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikolaevDMostDifVecNeighborsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikolaevDMostDifVecNeighborsPerfTests, kGtestValues, kPerfTestName);

}  // namespace nikolaev_d_most_dif_vec_neighbors
