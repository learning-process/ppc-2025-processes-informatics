#include <gtest/gtest.h>

#include "konstantinov_s_elem_vec_sign_change_count/common/include/common.hpp"
#include "konstantinov_s_elem_vec_sign_change_count/mpi/include/ops_mpi.hpp"
#include "konstantinov_s_elem_vec_sign_change_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

class KonstantinovSElemVecSignChangeTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(KonstantinovSElemVecSignChangeTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KonstantinovSElemVecSignChangeMPI, KonstantinovSElemVecSignChangeSEQ>(
        PPC_SETTINGS_konstantinov_s_elem_vec_sign_change_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KonstantinovSElemVecSignChangeTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KonstantinovSElemVecSignChangeTests, kGtestValues, kPerfTestName);

}  // namespace konstantinov_s_elem_vec_sign_change_count
