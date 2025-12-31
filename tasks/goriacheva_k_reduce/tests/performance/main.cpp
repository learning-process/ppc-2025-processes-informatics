#include <gtest/gtest.h>

#include "goriacheva_k_reduce/common/include/common.hpp"
#include "goriacheva_k_reduce/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_reduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace goriacheva_k_reduce {

class GoriachevaKReducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType GetTestInputData() final {
    return {10'000'000, 1};
  }

  bool CheckTestOutputData(OutType &output) final {
    return !output.empty();
  }
};

TEST_P(GoriachevaKReducePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    GoriachevaKReducePerfomanceTests, GoriachevaKReducePerfTests,
    ppc::util::TupleToGTestValues(ppc::util::MakeAllPerfTasks<InType, GoriachevaKReduceMPI, GoriachevaKReduceSEQ>(
        PPC_SETTINGS_goriacheva_k_reduce)),
    GoriachevaKReducePerfTests::CustomPerfTestName);

}  // namespace goriacheva_k_reduce
