#include <gtest/gtest.h>

#include "telnov_strongin_algorithm/common/include/common.hpp"
#include "telnov_strongin_algorithm/mpi/include/ops_mpi.hpp"
#include "telnov_strongin_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace telnov_strongin_algorithm {

class TelnovStronginAlgorithmPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    input_data_.a = 0.0;
    input_data_.b = 2.0;
    input_data_.eps = 1e-3;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == 1;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TelnovStronginAlgorithmPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TelnovStronginAlgorithmMPI, TelnovStronginAlgorithmSEQ>(
    PPC_SETTINGS_telnov_strongin_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TelnovStronginAlgorithmPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TelnovStronginAlgorithmPerfTests, kGtestValues, kPerfTestName);

}  // namespace telnov_strongin_algorithm
