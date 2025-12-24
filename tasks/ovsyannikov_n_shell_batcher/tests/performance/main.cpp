#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
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
    std::iota(input_data_.begin(), input_data_.end(), 0);
    
    // Фиксированный seed(42) для всех процессов
    std::mt19937 gen(42);
    std::shuffle(input_data_.begin(), input_data_.end(), gen);
    
    expected_output_ = input_data_;
    std::sort(expected_output_.begin(), expected_output_.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);
    if (is_initialized != 0) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Ваша проверка: Rank 1-3 просто подтверждают успех
    if (rank != 0) {
      return true;
    }

    // Реальная проверка только на Rank 0
    if (output_data.size() != expected_output_.size()) return false;
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(static_cast<double>(output_data[i]) - static_cast<double>(expected_output_[i])) > 1e-6) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final { return input_data_; }
 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(OvsyannikovNShellBatcherPerfTest, RunPerfModes) { ExecuteTest(GetParam()); }

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, OvsyannikovNShellBatcherMPI, OvsyannikovNShellBatcherSEQ>(
    PPC_SETTINGS_ovsyannikov_n_shell_batcher);

INSTANTIATE_TEST_SUITE_P(ovsyannikov_n_shell_batcher, OvsyannikovNShellBatcherPerfTest,
                         ppc::util::TupleToGTestValues(kAllPerfTasks),
                         [](const testing::TestParamInfo<OvsyannikovNShellBatcherPerfTest::ParamType> &info) {
                           return OvsyannikovNShellBatcherPerfTest::CustomPerfTestName(info);
                         });
}  // namespace ovsyannikov_n_shell_batcher