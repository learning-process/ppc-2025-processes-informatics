#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_global_search {

struct ExpectedSolution {
  std::vector<double> argmins;  // допустимые координаты минимума
  double value = 0.0;           // эталонное минимальное значение
  bool check_argmin = false;    // проверять ли координату
};

struct LocalTestCase {
  std::string name;
  Problem problem;
  ExpectedSolution expected;
};

using FuncParam = ppc::util::FuncTestParam<InType, OutType, LocalTestCase>;

class SizovDRunFuncTestsGlobalSearch : public ppc::util::BaseRunFuncTests<InType, OutType, LocalTestCase> {
 public:
  static std::string PrintTestParam(const LocalTestCase &tc) {
    return tc.name;
  }

  void PrepareTestCase(const FuncParam &param) {
    test_case_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param);
    input_ = test_case_.problem;
    expected_ = test_case_.expected;
  }

 protected:
  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {
    // 0) Алгоритм должен хотя бы выполняться
    if (out.iterations <= 0) {
      return false;
    }

    // converged — диагностический флаг, не критерий корректности
    (void)out.converged;

    // 1) Значения должны быть конечны
    if (!std::isfinite(out.value) || !std::isfinite(out.argmin)) {
      return false;
    }

    // 2) Аргумент должен лежать в допустимой области
    const double x_tol = std::max(10.0 * input_.accuracy, 1e-9);
    if (out.argmin < input_.left - x_tol || out.argmin > input_.right + x_tol) {
      return false;
    }

    // 3) Проверка значения функции (главная)
    const double f_tol = 50.0 * input_.accuracy;
    if (std::abs(out.value - expected_.value) > f_tol) {
      return false;
    }

    // 4) Проверка координаты — только когда минимум уникален
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
  LocalTestCase test_case_{};
  InType input_{};
  ExpectedSolution expected_{};
};

namespace {

constexpr double kPi = std::numbers::pi;

std::vector<LocalTestCase> GetTestCases() {
  std::vector<LocalTestCase> cases;
  cases.reserve(22U);

  cases.push_back({"case_001_linear_inc", Problem{[](double x) {
    return x;
  }, -5.0, 5.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{-5.0}, -5.0, true}});

  cases.push_back({"case_002_linear_dec", Problem{[](double x) {
    return -2.0 * x + 3.0;
  }, -5.0, 5.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{5.0}, -7.0, true}});

  cases.push_back({"case_003_quad_center", Problem{[](double x) {
    return x * x;
  }, -2.0, 2.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, 0.0, true}});

  cases.push_back({"case_004_quad_shifted_1", Problem{[](double x) {
    return (x - 1.0) * (x - 1.0);
  }, -1.5, 1.5, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{1.0}, 0.0, true}});

  cases.push_back({"case_005_quad_plus_5", Problem{[](double x) {
    return x * x + 5.0;
  }, -1.0, 1.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, 5.0, true}});

  cases.push_back({"case_006_abs", Problem{[](double x) {
    return std::abs(x);
  }, -4.0, 4.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, 0.0, true}});

  cases.push_back({"case_007_abs_shift_1", Problem{[](double x) {
    return std::abs(x - 1.0);
  }, -4.0, 4.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{1.0}, 0.0, true}});

  cases.push_back({"case_008_abs_plus_quad", Problem{[](double x) {
    return std::abs(x) + 0.1 * x * x;
  }, -1.0, 1.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, 0.0, true}});

  cases.push_back({"case_009_quad_mixed", Problem{[](double x) {
    return 0.2 * x * x - 0.3 * x;
  }, -1.5, 1.5, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.75}, -0.1125, true}});

  cases.push_back({"case_010_quartic", Problem{[](double x) {
    return x * x * x * x;
  }, -0.5, 0.5, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, 0.0, true}});

  cases.push_back({"case_011_exp_quad", Problem{[](double x) {
    return std::exp(x * x);
  }, -2.0, 2.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, 1.0, true}});

  cases.push_back({"case_012_log_quad", Problem{[](double x) {
    return std::log(x * x + 2.0);
  }, -2.0, 2.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, std::log(2.0), true}});

  cases.push_back({"case_013_sqrt_quad", Problem{[](double x) {
    return std::sqrt(x * x + 1.0);
  }, -3.0, 3.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, 1.0, true}});

  cases.push_back({"case_014_abs_plus_exp_quad", Problem{[](double x) {
    return std::abs(x) + std::exp(x * x);
  }, -2.0, 2.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, 1.0, true}});

  cases.push_back({"case_015_log_sqrt_combo", Problem{[](double x) {
    return std::log(std::sqrt(x * x + 1.0) + 1.0);
  }, -3.0, 3.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{0.0}, std::log(2.0), true}});

  cases.push_back({"case_016_exp_shift_1", Problem{[](double x) {
    return std::exp((x - 1.0) * (x - 1.0));
  }, -1.0, 3.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{1.0}, 1.0, true}});

  cases.push_back({"case_017_log_abs_shift_1", Problem{[](double x) {
    return std::log(std::abs(x - 1.0) + 2.0);
  }, -2.0, 4.0, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{1.0}, std::log(2.0), true}});

  cases.push_back({"case_018_sin_sq", Problem{[](double x) {
    double s = std::sin(x);
    return s * s;
  }, -kPi, kPi, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{-kPi, 0.0, kPi}, 0.0, false}});

  cases.push_back({"case_019_exp_sin_sq", Problem{[](double x) {
    return std::exp(std::sin(x) * std::sin(x));
  }, -kPi, kPi, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{-kPi, 0.0, kPi}, 1.0, false}});

  cases.push_back({"case_020_sqrt_sin_sq_plus_1", Problem{[](double x) {
    return std::sqrt(std::sin(x) * std::sin(x) + 1.0);
  }, -kPi, kPi, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{-kPi, 0.0, kPi}, 1.0, false}});

  cases.push_back({"case_021_cos_on_segment", Problem{[](double x) {
    return std::cos(x);
  }, -kPi, kPi, kDefaultAccuracy, kDefaultReliability, kDefaultMaxIterations},
                   ExpectedSolution{{-kPi, kPi}, -1.0, false}});

  return cases;
}

std::vector<FuncParam> BuildTestTasks(const std::vector<LocalTestCase> &tests) {
  std::vector<FuncParam> tasks;
  tasks.reserve(tests.size() * 2U);

  const std::string mpi_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchMPI>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchMPI::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  const std::string seq_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchSEQ>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchSEQ::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  for (const auto &t : tests) {
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchMPI, InType>, mpi_name, t);
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchSEQ, InType>, seq_name, t);
  }

  return tasks;
}

class FuncTestInstance final : public SizovDRunFuncTestsGlobalSearch {
 public:
  explicit FuncTestInstance(FuncParam param) : param_(std::move(param)) {}

  void TestBody() override {
    PrepareTestCase(param_);
    ExecuteTest(param_);
  }

 private:
  FuncParam param_;
};

struct FuncTestRegistrar {
  explicit FuncTestRegistrar(std::vector<FuncParam> tasks) : tasks_(std::move(tasks)) {
    test_names_.reserve(tasks_.size());

    std::size_t idx = 0;
    for (const auto &param : tasks_) {
      const auto &name_test = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(param);
      const auto &tc = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param);

      const std::string test_name = name_test + "_" + SizovDRunFuncTestsGlobalSearch::PrintTestParam(tc);
      test_names_.push_back(test_name);

      const std::size_t current_idx = idx++;
      ::testing::RegisterTest(
          "SizovDGlobalSearchFuncTests", test_names_.back().c_str(), nullptr, nullptr, __FILE__, __LINE__,
          [this, current_idx]() -> ::testing::Test * { return new FuncTestInstance(tasks_[current_idx]); });
    }
  }

 private:
  std::vector<FuncParam> tasks_;
  std::vector<std::string> test_names_;
};

const bool kRegisteredFuncTests = []() {
  const auto cases = GetTestCases();
  const auto tasks = BuildTestTasks(cases);
  static const FuncTestRegistrar registrar(tasks);
  (void)registrar;
  return true;
}();

}  // namespace
}  // namespace sizov_d_global_search
