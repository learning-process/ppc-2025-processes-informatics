#include <gtest/gtest.h>

#include "telnov_transfer_one_all/common/include/common.hpp"
#include "telnov_transfer_one_all/mpi/include/ops_mpi.hpp"
#include "telnov_transfer_one_all/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace telnov_transfer_one_all {

class TelnovTransferOneAllPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    const int array_size = 1000000;
    input_data_.resize(array_size, 42);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::equal(input_data_.begin(), input_data_.end(), output_data.begin());
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TelnovTransferOneAllPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks(
  std::make_tuple(
    std::make_tuple(ppc::task::TaskGetter<TelnovTransferOneAllMPI<int>, InType>, "MPI"),
      std::make_tuple(ppc::task::TaskGetter<TelnovTransferOneAllSEQ<int>, InType>, "SEQ")
  ),
  PPC_SETTINGS_telnov_transfer_one_all);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TelnovTransferOneAllPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TelnovTransferOneAllPerfTests, kGtestValues, kPerfTestName);

}  // namespace telnov_transfer_one_all
