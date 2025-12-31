#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "egashin_k_radix_batcher_sort/common/include/common.hpp"
#include "egashin_k_radix_batcher_sort/mpi/include/ops_mpi.hpp"
#include "egashin_k_radix_batcher_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace egashin_k_radix_batcher_sort {

class EgashinKRunPerfTestRadixBatcherSort : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1000000;
  InType input_data_{};

  void SetUp() override {
    std::mt19937 gen(42);  // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution<double> dist(-1e6, 1e6);

    input_data_.resize(kCount_);
    for (int i = 0; i < kCount_; ++i) {
      input_data_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    OutType expected_data = input_data_;
    std::ranges::sort(expected_data);
    if (output_data.size() != expected_data.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_data[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(EgashinKRunPerfTestRadixBatcherSort, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, EgashinKRadixBatcherSortMPI, EgashinKRadixBatcherSortSEQ>(PPC_SETTINGS_egashin_k_radix_batcher_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = EgashinKRunPerfTestRadixBatcherSort::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, EgashinKRunPerfTestRadixBatcherSort, kGtestValues, kPerfTestName);

}  // namespace egashin_k_radix_batcher_sort
