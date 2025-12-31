#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"
#include "eremin_v_strongin_algorithm/mpi/include/ops_mpi.hpp"
#include "eremin_v_strongin_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace eremin_v_strongin_algorithm {

class EreminVRunPerfTestsStronginAlgorithm : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    double lower_bound_ = 0.0;
    double upper_bound_ = 10.0;
    int steps_ = 100000000;
    input_data_ = std::make_tuple(lower_bound_, upper_bound_, steps_, Function);
    expected_result_ = FindMinimum(Function, lower_bound_, upper_bound_, 1e-3);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double tolerance = std::max(std::abs(expected_result_) * 0.01, 1e-8);
    return std::abs(output_data - expected_result_) <= tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_{};

  static double Function(double x) {
    return (x * x * std::exp(x) * std::sin(x)) + (x * x * x * x * std::cos(2 * x));
  }

  static double FindMinimum(const std::function<double(double)> &f, double a, double b, double step) {
    double min_val = std::numeric_limits<double>::infinity();
    for (double x = a; x <= b; x += step) {
      double fx = f(x);
      if (fx < min_val) {
        min_val = fx;
      }
    }
    return min_val;
  }
};

TEST_P(EreminVRunPerfTestsStronginAlgorithm, RunPerfModesStronginAlgorithm) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, EreminVStronginAlgorithmMPI, EreminVStronginAlgorithmSEQ>(
        PPC_SETTINGS_eremin_v_strongin_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = EreminVRunPerfTestsStronginAlgorithm::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTestsStronginAlgorithm, EreminVRunPerfTestsStronginAlgorithm, kGtestValues,
                         kPerfTestName);

}  // namespace eremin_v_strongin_algorithm
