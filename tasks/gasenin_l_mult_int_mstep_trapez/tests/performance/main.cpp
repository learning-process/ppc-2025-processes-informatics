#include <gtest/gtest.h>
#include <mpi.h>

#include "gasenin_l_mult_int_mstep_trapez/common/include/common.hpp"
#include "gasenin_l_mult_int_mstep_trapez/mpi/include/ops_mpi.hpp"
#include "gasenin_l_mult_int_mstep_trapez/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
namespace gasenin_l_mult_int_mstep_trapez {

class GaseninLRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = TaskData{2000, 1, 0.0, 1.0, 0.0, 1.0};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    double expected = 2.0 / 3.0;
    double tolerance = 1e-3;
    return std::abs(output_data - expected) < tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(GaseninLRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GaseninLMultIntMstepTrapezMPI, GaseninLMultIntMstepTrapezSEQ>(
        PPC_SETTINGS_gasenin_l_mult_int_mstep_trapez);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = GaseninLRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(GaseninLIntegralPerfTests, GaseninLRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace gasenin_l_mult_int_mstep_trapez
