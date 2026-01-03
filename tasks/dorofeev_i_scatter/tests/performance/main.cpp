#include <gtest/gtest.h>

#include "dorofeev_i_scatter/common/include/common.hpp"
#include "dorofeev_i_scatter/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_scatter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorofeev_i_scatter {

class DorofeevIScatterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_.resize(100000000);
    std::fill(input_.begin(), input_.end(), 1.0);
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    return std::abs(output_data - 100000000.0) < 1e-6;
  }

 private:
  InType input_;
};

TEST_P(DorofeevIScatterPerfTests, ScatterPerf) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DorofeevIScatterMPI, DorofeevIScatterSEQ>(PPC_SETTINGS_dorofeev_i_scatter);

INSTANTIATE_TEST_SUITE_P(ScatterPerf, DorofeevIScatterPerfTests, ppc::util::TupleToGTestValues(kPerfTasks),
                         DorofeevIScatterPerfTests::CustomPerfTestName);

}  // namespace dorofeev_i_scatter
