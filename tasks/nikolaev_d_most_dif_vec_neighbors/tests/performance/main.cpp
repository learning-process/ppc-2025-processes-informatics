#include <gtest/gtest.h>

  #include <random>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/mpi/include/ops_mpi.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

class NikolaevDMostDifVecNeighborsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t kCount_ = 10'000'000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_.resize(kCount_);

    //std::random_device dev;
    std::mt19937 rng(42); //dev() 42
    std::uniform_int_distribution<int> dist(-10'000, 10'000);

    for (size_t i = 0; i < kCount_; ++i) {
      input_data_[i] = dist(rng);
    }

    std::uniform_int_distribution<size_t> pos_dist(0, kCount_ - 2);
    size_t max_dif_pos = pos_dist(rng);

    input_data_[max_dif_pos] = -15'000;
    input_data_[max_dif_pos + 1] = 15'000;
    expected_output_ = std::make_pair(-15'000, 15'000);

    // for (int i=0; i < 10; i++) {
    //   std::cout << "input_data_[" << i << "] = " << input_data_[i] << "\n";
    // }
    // std::cout << "max_dif_pos=" << max_dif_pos << "\n";
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
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
