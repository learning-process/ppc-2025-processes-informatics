#include <gtest/gtest.h>

//#include <cstddef>
//#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/mpi/include/ops_mpi.hpp"
#include "nikolaev_d_most_dif_vec_neighbors/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

class NikolaevDMostDifVecNeighborsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  //const size_t kCount_ = 10'000'000;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    // input_data_.resize(kCount_);

    // std::random_device dev;
    // std::mt19937 rng(dev());
    // std::uniform_int_distribution<int> dist(-10'000, 10'000);

    // for (size_t i = 0; i < kCount_; ++i) {
    //   input_data_[i] = dist(rng);
    // }

    // std::uniform_int_distribution<size_t> pos_dist(0, kCount_ - 2);
    // size_t max_dif_pos = pos_dist(rng);

    // input_data_[max_dif_pos] = -15'000;
    // input_data_[max_dif_pos + 1] = 15'000;
    // expected_output_ = std::make_pair(-15'000, 15'000);
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_nikolaev_d_most_dif_vec_neighbors, "test_vec.txt");
    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("error");
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    int number = 0;
    while (iss >> number) {
      input_data_.push_back(number);
    }

    file.close();

    expected_output_ = std::make_pair(-15'000, 15'000);
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
