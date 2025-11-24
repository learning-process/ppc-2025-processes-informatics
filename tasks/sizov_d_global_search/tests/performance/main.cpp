#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sizov_d_global_search {

namespace {

constexpr double kLeft = -10.0;
constexpr double kRight = 10.0;
constexpr double kAccuracy = 1e-4;
constexpr double kReliability = 3.0;
constexpr int kMaxIterations = 80000;

// Ограничитель на «раздувание» значения в тяжёлой функции,
// чтобы избежать NaN/inf при случайных переполнениях.
constexpr double kValuePenalty = 1.0e6;

InType MakePerfProblem() {
  InType problem{};

  problem.left = kLeft;
  problem.right = kRight;
  problem.accuracy = kAccuracy;
  problem.reliability = kReliability;
  problem.max_iterations = kMaxIterations;

  problem.func = [](double x) {
    constexpr double a1 = 3.5;
    constexpr double a2 = 1.7;
    constexpr double b1 = 0.35;
    constexpr double c1 = 0.01;
    constexpr double d1 = 0.15;
    constexpr double d2 = 0.05;
    constexpr double e1 = 0.02;
    constexpr double f1 = 0.001;

    auto clamp_value = [](double v) {
      if (!std::isfinite(v)) {
        return kValuePenalty;
      }
      if (std::abs(v) > kValuePenalty) {
        return (v > 0.0 ? kValuePenalty : -kValuePenalty);
      }
      return v;
    };

    double value = std::sin(a1 * x) + b1 * std::cos(a2 * x) + c1 * x * x;
    value = clamp_value(value);

    for (int i = 0; i < 25; ++i) {
      value = std::sin(value + d1 * x) + std::cos(value - d2 * x) + e1 * value * value;
      value = clamp_value(value);
    }

    value += f1 * x * x * x;
    value = clamp_value(value);

    return value;
  };

  return problem;
}

}  // namespace

class SizovDGlobalSearchPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    input_data_ = MakePerfProblem();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для perf-теста достаточно:
    // 1) алгоритм сошёлся;
    // 2) значение конечное;
    // 3) найденный аргумент в допустимом интервале (с небольшим допуском).
    if (!output_data.converged) {
      return false;
    }
    if (!std::isfinite(output_data.value)) {
      return false;
    }

    const double x = output_data.argmin;
    const double eps = 1e-6;
    if (x < input_data_.left - eps || x > input_data_.right + eps) {
      return false;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  InType input_data_{};
};

TEST_P(SizovDGlobalSearchPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SizovDGlobalSearchMPI, SizovDGlobalSearchSEQ>(
    PPC_SETTINGS_sizov_d_global_search);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SizovDGlobalSearchPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SizovDGlobalSearchPerfTests, kGtestValues, kPerfTestName);

}  // namespace sizov_d_global_search
