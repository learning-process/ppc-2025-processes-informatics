#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"
#include "frolova_s_mult_int_trapez/mpi/include/ops_mpi.hpp"
#include "frolova_s_mult_int_trapez/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace frolova_s_mult_int_trapez {

class FrolovaRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType small_input_{};
  InType medium_input_{};
  InType large_input_{};

  void SetUp() override {
    // Small problem
    small_input_ = {.limits = {{0.0, 1.0}, {4.0, 6.0}},
                    .number_of_intervals = {100, 100},
                    .function = [](std::vector<double> input) {
      return (-3.0 * std::pow(input[1], 2) * std::sin(5.0 * input[0])) / 2.0;
    }};

    // Medium problem
    medium_input_ = {.limits = {{0.0, 1.0}, {4.0, 6.0}},
                     .number_of_intervals = {300, 300},
                     .function = [](std::vector<double> input) {
      return (-3.0 * std::pow(input[1], 2) * std::sin(5.0 * input[0])) / 2.0;
    }};

    // Large problem
    large_input_ = {
        .limits = {{0.0, 2.0}, {0.0, 2.0}},
        .number_of_intervals = {500, 500},
        .function = [](std::vector<double> input) { return std::pow(input[0], 3) + std::pow(input[1], 3); }};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::isfinite(output_data);
  }

  InType GetTestInputData() final {
    auto test_param = GetParam();

    std::string test_name = std::get<1>(test_param);

    if (test_name.find("small") != std::string::npos) {
      return small_input_;
    } else if (test_name.find("medium") != std::string::npos) {
      return medium_input_;
    } else if (test_name.find("large") != std::string::npos) {
      return large_input_;
    }

    return small_input_;
  }
};

TEST_P(FrolovaRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, FrolovaSMultIntTrapezMPI, FrolovaSMultIntTrapezSEQ>(
    PPC_SETTINGS_frolova_s_mult_int_trapez);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FrolovaRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(TrapezoidalIntegrationPerfTests, FrolovaRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace frolova_s_mult_int_trapez
