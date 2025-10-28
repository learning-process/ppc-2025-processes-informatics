#include <gtest/gtest.h>

#include "lukin_i_elem_vec_sum/common/include/common.hpp"
#include "lukin_i_elem_vec_sum/mpi/include/ops_mpi.hpp"
#include "lukin_i_elem_vec_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lukin_i_elem_vec_sum {

class LukinIRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int vec_size = 10000;
  const int vec_value = 1;

  InType input_data_;
  OutType expected_result;

  void SetUp() override {
    input_data_ = std::vector<int>(vec_size, vec_value);
    expected_result = vec_size * vec_value;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_result == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LukinIRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LukinIElemVecSumMPI, LukinIElemVecSumSEQ>(PPC_SETTINGS_lukin_i_elem_vec_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LukinIRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunTests, LukinIRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace lukin_i_elem_vec_sum
