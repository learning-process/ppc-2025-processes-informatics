#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>

#include "dolov_v_qsort_batcher/common/include/common.hpp"
#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"
#include "dolov_v_qsort_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dolov_v_qsort_batcher {

class DolovVQsortBatcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data_{};

  void SetUp() override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      const int kCount = 10000000;
      input_data_.resize(kCount);
      for (int i = 0; i < kCount; ++i) {
        input_data_[i] = (i % 2 == 0) ? static_cast<double>(i) : static_cast<double>(kCount - i);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      return std::is_sorted(output_data.begin(), output_data.end());
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DolovVQsortBatcherPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DolovVQsortBatcherMPI, DolovVQsortBatcherSEQ>(
    PPC_SETTINGS_dolov_v_qsort_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DolovVQsortBatcherPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DolovVQsortBatcherPerfTests, kGtestValues, kPerfTestName);

}  // namespace dolov_v_qsort_batcher
