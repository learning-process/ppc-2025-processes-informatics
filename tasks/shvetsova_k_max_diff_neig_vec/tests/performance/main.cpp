#include <gtest/gtest.h>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "shvetsova_k_max_diff_neig_vec/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_max_diff_neig_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shvetsova_k_max_diff_neig_vec {

class ShvetsovaKMaxDiffNeigVecRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(ShvetsovaKMaxDiffNeigVecRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShvetsovaKMaxDiffNeigVecMPI, ShvetsovaKMaxDiffNeigVecSEQ>(
        PPC_SETTINGS_shvetsova_k_max_diff_neig_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShvetsovaKMaxDiffNeigVecRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShvetsovaKMaxDiffNeigVecRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace shvetsova_k_max_diff_neig_vec
