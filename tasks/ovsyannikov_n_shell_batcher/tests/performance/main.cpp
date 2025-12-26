#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include "ovsyannikov_n_shell_batcher/common/include/common.hpp"
#include "ovsyannikov_n_shell_batcher/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ovsyannikov_n_shell_batcher {

class OvsyannikovNShellBatcherPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int size = 40000;
    input_data_.resize(size);
    std::ranges::iota(input_data_, 0);
    std::mt19937 gen(42);  // NOLINT(cert-msc51-cpp)
    std::ranges::shuffle(input_data_, gen);
    expected_output_ = input_data_;
    std::ranges::sort(expected_output_);
  }
  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
    return output_data == expected_output_;
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(OvsyannikovNShellBatcherPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ovsyannikov_n_shell_batcher_mpi, OvsyannikovNShellBatcherPerfTest,
    ppc::util::TupleToGTestValues(ppc::util::MakePerfTaskTuples<OvsyannikovNShellBatcherMPI, InType>("mpi")),
    [](const testing::TestParamInfo<OvsyannikovNShellBatcherPerfTest::ParamType> &info) {
      return OvsyannikovNShellBatcherPerfTest::CustomPerfTestName(info);
    });

INSTANTIATE_TEST_SUITE_P(
    ovsyannikov_n_shell_batcher_seq, OvsyannikovNShellBatcherPerfTest,
    ppc::util::TupleToGTestValues(ppc::util::MakePerfTaskTuples<OvsyannikovNShellBatcherSEQ, InType>("seq")),
    [](const testing::TestParamInfo<OvsyannikovNShellBatcherPerfTest::ParamType> &info) {
      return OvsyannikovNShellBatcherPerfTest::CustomPerfTestName(info);
    });
}  // namespace ovsyannikov_n_shell_batcher
