#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_global_search {

// -----------------------------------------------------------------------------
// Test data
// -----------------------------------------------------------------------------

struct ExpectedSolution {
  std::vector<double> argmins;
  double value = 0.0;
  bool check_argmin = false;
};

struct LocalTestCase {
  std::string name;
  Problem problem;
  ExpectedSolution expected;
};

using FuncParam = ppc::util::FuncTestParam<InType, OutType, LocalTestCase>;

// -----------------------------------------------------------------------------
// Test fixture
// -----------------------------------------------------------------------------

class SizovDRunFuncTestsGlobalSearch : public ppc::util::BaseRunFuncTests<InType, OutType, LocalTestCase> {
 public:
  static std::string PrintTestParam(const LocalTestCase &tc) {
    return ppc::util::test::SanitizeToken(tc.name);
  }

 protected:
  void SetUp() override {
    const auto &tc = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = tc.problem;
    expected_ = tc.expected;
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {
    if (out.iterations <= 0) {
      return false;
    }

    // converged — диагностический флаг, не критерий успеха
    (void)out.converged;

    if (!std::isfinite(out.value) || !std::isfinite(out.argmin)) {
      return false;
    }

    const double x_tol = std::max(10.0 * input_.accuracy, 1e-9);
    if (out.argmin < input_.left - x_tol || out.argmin > input_.right + x_tol) {
      return false;
    }

    const double f_tol = 50.0 * input_.accuracy;
    if (std::abs(out.value - expected_.value) > f_tol) {
      return false;
    }

    if (expected_.check_argmin) {
      double best_dx = std::numeric_limits<double>::infinity();
      for (double a : expected_.argmins) {
        best_dx = std::min(best_dx, std::abs(out.argmin - a));
      }

      const double arg_tol = 20.0 * input_.accuracy;
      if (best_dx > arg_tol) {
        return false;
      }
    }

    return true;
  }

 private:
  InType input_{};
  ExpectedSolution expected_{};
};

// -----------------------------------------------------------------------------
// Test cases
// -----------------------------------------------------------------------------

namespace {

constexpr double kPi = std::numbers::pi;

std::vector<LocalTestCase> GetTestCases() {
  std::vector<LocalTestCase> cases;
  cases.reserve(21);

  // 001: линейная возрастает => минимум на left
  cases.push_back(
      {"linear_inc",
       Problem{[](double x) { return x; }, -5.0, 5.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
       {{-5.0}, -5.0, true}});

  // 002: линейная убывает => минимум на right
  cases.push_back({"linear_dec",
                   Problem{[](double x) { return -2.0 * x + 3.0; }, -5.0, 5.0, kDefaultAccuracy, kDefaultReliability,
                           kDefaultMaxIterations},
                   {{5.0}, -7.0, true}});

  // 003: x^2 => минимум в 0
  cases.push_back(
      {"quad_center",
       Problem{[](double x) { return x * x; }, -2.0, 2.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
       {{0.0}, 0.0, true}});

  // 004: (x - 1)^2 => минимум в 1
  cases.push_back({"quad_shift_1",
                   Problem{[](double x) { return (x - 1.0) * (x - 1.0); }, -1.5, 1.5, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{1.0}, 0.0, true}});

  // 005: x^2 + 5
  cases.push_back({"quad_plus_5",
                   Problem{[](double x) { return x * x + 5.0; }, -1.0, 1.0, kDefaultAccuracy, kDefaultReliability,
                           kDefaultMaxIterations},
                   {{0.0}, 5.0, true}});

  // 006: |x|
  cases.push_back({"abs",
                   Problem{[](double x) { return std::abs(x); }, -4.0, 4.0, kDefaultAccuracy, kDefaultReliability,
                           kDefaultMaxIterations},
                   {{0.0}, 0.0, true}});

  // 007: |x - 1|
  cases.push_back({"abs_shift_1",
                   Problem{[](double x) { return std::abs(x - 1.0); }, -4.0, 4.0, kDefaultAccuracy, kDefaultReliability,
                           kDefaultMaxIterations},
                   {{1.0}, 0.0, true}});

  // 008: |x| + 0.1 x^2
  cases.push_back({"abs_plus_quad",
                   Problem{[](double x) { return std::abs(x) + 0.1 * x * x; }, -1.0, 1.0, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{0.0}, 0.0, true}});

  // 009: 0.2x^2 - 0.3x
  cases.push_back({"quad_mixed",
                   Problem{[](double x) { return 0.2 * x * x - 0.3 * x; }, -1.5, 1.5, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{0.75}, -0.1125, true}});

  // 010: x^4
  cases.push_back({"quartic",
                   Problem{[](double x) { return x * x * x * x; }, -0.5, 0.5, kDefaultAccuracy, kDefaultReliability,
                           kDefaultMaxIterations},
                   {{0.0}, 0.0, true}});

  // 011: exp(x^2)
  cases.push_back({"exp_quad",
                   Problem{[](double x) { return std::exp(x * x); }, -2.0, 2.0, kDefaultAccuracy, kDefaultReliability,
                           kDefaultMaxIterations},
                   {{0.0}, 1.0, true}});

  // 012: log(x^2 + 2)
  cases.push_back({"log_quad",
                   Problem{[](double x) { return std::log(x * x + 2.0); }, -2.0, 2.0, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{0.0}, std::log(2.0), true}});

  // 013: sqrt(x^2 + 1)
  cases.push_back({"sqrt_quad",
                   Problem{[](double x) { return std::sqrt(x * x + 1.0); }, -3.0, 3.0, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{0.0}, 1.0, true}});

  // 014: |x| + exp(x^2)
  cases.push_back({"abs_plus_exp_quad",
                   Problem{[](double x) { return std::abs(x) + std::exp(x * x); }, -2.0, 2.0, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{0.0}, 1.0, true}});

  // 015: log(sqrt(x^2 + 1) + 1)
  cases.push_back({"log_sqrt_combo",
                   Problem{[](double x) { return std::log(std::sqrt(x * x + 1.0) + 1.0); }, -3.0, 3.0, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{0.0}, std::log(2.0), true}});

  // 016: exp((x - 1)^2)
  cases.push_back({"exp_shift_1",
                   Problem{[](double x) { return std::exp((x - 1.0) * (x - 1.0)); }, -1.0, 3.0, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{1.0}, 1.0, true}});

  // 017: log(|x - 1| + 2)
  cases.push_back({"log_abs_shift_1",
                   Problem{[](double x) { return std::log(std::abs(x - 1.0) + 2.0); }, -2.0, 4.0, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{1.0}, std::log(2.0), true}});

  // 018: sin^2(x) — множество минимумов
  cases.push_back({"sin_sq",
                   Problem{[](double x) {
    const double s = std::sin(x);
    return s * s;
  }, -kPi, kPi, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   {{-kPi, 0.0, kPi}, 0.0, false}});

  // 019: exp(sin^2(x))
  cases.push_back({"exp_sin_sq",
                   Problem{[](double x) { return std::exp(std::sin(x) * std::sin(x)); }, -kPi, kPi, kDefaultAccuracy,
                           kDefaultReliability, kDefaultMaxIterations},
                   {{-kPi, 0.0, kPi}, 1.0, false}});

  // 020: sqrt(sin^2(x) + 1)
  cases.push_back({"sqrt_sin_sq_plus_1",
                   Problem{[](double x) { return std::sqrt(std::sin(x) * std::sin(x) + 1.0); }, -kPi, kPi,
                           kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   {{-kPi, 0.0, kPi}, 1.0, false}});

  // 021: cos(x) — минимум на границах
  cases.push_back({"cos_on_segment",
                   Problem{[](double x) { return std::cos(x); }, -kPi, kPi, kDefaultAccuracy, kDefaultReliability,
                           kDefaultMaxIterations},
                   {{-kPi, kPi}, -1.0, false}});

  return cases;
}

// -----------------------------------------------------------------------------
// Task list builder
// -----------------------------------------------------------------------------

std::vector<FuncParam> BuildTestTasks(const std::vector<LocalTestCase> &tests) {
  std::vector<FuncParam> tasks;
  tasks.reserve(tests.size() * 2);

  for (const auto &t : tests) {
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchMPI, InType>,
                       std::string(ppc::util::GetNamespace<SizovDGlobalSearchMPI>()) + "_" +
                           ppc::task::GetStringTaskType(SizovDGlobalSearchMPI::GetStaticTypeOfTask(),
                                                        PPC_SETTINGS_sizov_d_global_search),
                       t);

    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchSEQ, InType>,
                       std::string(ppc::util::GetNamespace<SizovDGlobalSearchSEQ>()) + "_" +
                           ppc::task::GetStringTaskType(SizovDGlobalSearchSEQ::GetStaticTypeOfTask(),
                                                        PPC_SETTINGS_sizov_d_global_search),
                       t);
  }

  return tasks;
}

// -----------------------------------------------------------------------------
// GTest instantiation
// -----------------------------------------------------------------------------

TEST_P(SizovDRunFuncTestsGlobalSearch, GlobalSearchWorks) {
  ExecuteTest(GetParam());
}

const auto kTestCases = GetTestCases();
const auto kTestTasks = BuildTestTasks(kTestCases);
const auto kGtestValues = ::testing::ValuesIn(kTestTasks);

const auto kTestName = SizovDRunFuncTestsGlobalSearch::PrintFuncTestName<SizovDRunFuncTestsGlobalSearch>;

INSTANTIATE_TEST_SUITE_P(SizovDGlobalSearch, SizovDRunFuncTestsGlobalSearch, kGtestValues, kTestName);

}  // namespace

}  // namespace sizov_d_global_search
