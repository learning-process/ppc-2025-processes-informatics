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

class EgashinKRadixBatcherSortPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    std::size_t arr_size = 1000000;
    std::mt19937 gen(42);  // NOLINT(cert-msc51-cpp)
    std::uniform_real_distribution<double> dist(-1e6, 1e6);

    input_.resize(arr_size);
    for (std::size_t i = 0; i < arr_size; ++i) {
      input_[i] = dist(gen);
    }

    expected_ = input_;
    std::ranges::sort(expected_);
  }

  bool CheckTestOutputData(OutType &output) override {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    if (output.size() != expected_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output.size(); ++i) {
      if (output[i] != expected_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() override {
    return input_;
  }

 private:
  InType input_;
  OutType expected_;
};

TEST_P(EgashinKRadixBatcherSortPerfTest, Performance) {
  ExecuteTest(GetParam());
}

const auto kPerfParams =
    ppc::util::MakeAllPerfTasks<InType, TestTaskMPI, TestTaskSEQ>(PPC_SETTINGS_egashin_k_radix_batcher_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kPerfParams);

const auto kPerfTestName = EgashinKRadixBatcherSortPerfTest::CustomPerfTestName;

// NOLINTNEXTLINE
INSTANTIATE_TEST_SUITE_P(EgashinKRadixBatcherSortPerf, EgashinKRadixBatcherSortPerfTest, kGtestValues, kPerfTestName);

}  // namespace egashin_k_radix_batcher_sort
