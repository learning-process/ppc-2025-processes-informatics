#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "makovskiy_i_allreduce/common/include/common.hpp"
#include "makovskiy_i_allreduce/mpi/include/ops_mpi.hpp"
#include "makovskiy_i_allreduce/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace makovskiy_i_allreduce {

class AllreducePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType GetTestInputData() final {
    constexpr int kCount = 100000000;
    return InType(kCount, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    constexpr int kCount = 100000000;
    return !output_data.empty() && output_data[0] == kCount;
  }
};

TEST_P(AllreducePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TestTaskSEQ, TestTaskMPI>(PPC_SETTINGS_makovskiy_i_allreduce);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = AllreducePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(AllreducePerf, AllreducePerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace makovskiy_i_allreduce
