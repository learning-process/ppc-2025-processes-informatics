#include <gtest/gtest.h>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"
#include "dorofeev_i_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dorofeev_i_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {
namespace {

class DorofeevIRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(DorofeevIRunPerfTestProcesses, DorofeevIRunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kkAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DorofeevIMonteCarloIntegrationMPI, DorofeevIMonteCarloIntegrationSEQ>(PPC_SETTINGS_example_processes);

const auto kkGtestValues = ppc::util::TupleToGTestValues(kkAllPerfTasks);

const auto kkPerfTestName = DorofeevIRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(DorofeevIRunModeTests, DorofeevIRunPerfTestProcesses, kkGtestValues, kkPerfTestName);

} // namespace
}  // namespace dorofeev_i_monte_carlo_integration_processes
