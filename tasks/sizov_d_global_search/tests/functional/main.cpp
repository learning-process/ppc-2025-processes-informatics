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
  cases.reserve(2);

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
