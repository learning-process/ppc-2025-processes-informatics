#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "lifanov_k_trapezoid_method/common/include/common.hpp"
#include "lifanov_k_trapezoid_method/mpi/include/ops_mpi.hpp"
#include "lifanov_k_trapezoid_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lifanov_k_trapezoid_method {

class LifanovKTrapezoidMethodPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data_;

  void SetUp() override {
    // Крупная сетка для perf-теста
    input_data_ = {0.0, 1.0, 0.0, 1.0, 2000.0, 2000.0};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
    return output_data > 0.0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LifanovKTrapezoidMethodPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LifanovKTrapezoidMethodMPI, LifanovKTrapezoidMethodSEQ>(PPC_SETTINGS_lifanov_k_trapezoid_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LifanovKTrapezoidMethodPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MyTrapezoidMethodPerfTests, LifanovKTrapezoidMethodPerfTests, kGtestValues, kPerfTestName);

}  // namespace lifanov_k_trapezoid_method
