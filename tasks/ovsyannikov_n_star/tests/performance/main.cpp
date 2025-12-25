#include <gtest/gtest.h>
#include <vector>
#include "ovsyannikov_n_star/common/include/common.hpp"
#include "ovsyannikov_n_star/mpi/include/ops_mpi.hpp"
#include "ovsyannikov_n_star/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace ovsyannikov_n_star {

class OvsyannikovNRunPerfTestStar : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = {0, 0, 1000}; 
    expected_output_ = 1000;
  }

  bool CheckTestOutputData(OutType &actual_res) final {
    return actual_res == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

TEST_P(OvsyannikovNRunPerfTestStar, RunPerfModes) {
  ExecuteTest(GetParam());
}

// Генерация задач
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, OvsyannikovNStarMPI, OvsyannikovNStarSEQ>(PPC_SETTINGS_ovsyannikov_n_star);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
INSTANTIATE_TEST_SUITE_P(StarPerfTests, OvsyannikovNRunPerfTestStar, kGtestValues, 
                         OvsyannikovNRunPerfTestStar::CustomPerfTestName);

}  // namespace ovsyannikov_n_star