#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>

#include "global_search_strongin/common/include/common.hpp"
#include "global_search_strongin/mpi/include/ops_mpi.hpp"
#include "global_search_strongin/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace global_search_strongin {

struct StronginCase {
  double left = 0.0;
  double right = 1.0;
  double epsilon = 1e-3;
  double reliability = 2.0;
  int max_iterations = 200;
  std::function<double(double)> objective;
};

using TestType = StronginCase;

class StronginFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    const auto to_gtest_name_int = [](double v) {
      const int iv = static_cast<int>(v);
      if (iv < 0) {
        return std::string("m") + std::to_string(-iv);
      }
      return std::to_string(iv);
    };

    const auto l = to_gtest_name_int(param.left);
    const auto r = to_gtest_name_int(param.right);
    const auto it = std::to_string(param.max_iterations);
    return l + "_" + r + "_" + it;
  }

 protected:
  void SetUp() override {
    const auto &param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_.left = param.left;
    input_.right = param.right;
    input_.epsilon = param.epsilon;
    input_.reliability = param.reliability;
    input_.max_iterations = param.max_iterations;
    input_.objective = param.objective;
    reference_min_ = ComputeReferenceMin(input_);
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!std::isfinite(output_data.best_point) || !std::isfinite(output_data.best_value)) {
      return false;
    }
    if (output_data.best_point < input_.left || output_data.best_point > input_.right) {
      return false;
    }
    if (output_data.iterations < 0 || output_data.iterations > input_.max_iterations) {
      return false;
    }

    const double f_at_best = input_.objective(output_data.best_point);
    if (!std::isfinite(f_at_best)) {
      return false;
    }
    if (std::fabs(f_at_best - output_data.best_value) > 1e-9) {
      return false;
    }

    // Allow some slack vs. dense sampling reference.
    return output_data.best_value <= reference_min_ + 1e-2;
  }

 private:
  static double ComputeReferenceMin(const InType &input) {
    constexpr std::size_t kSamples = 20000;
    const double step = (input.right - input.left) / static_cast<double>(kSamples);
    double best = input.objective(input.left);
    for (std::size_t i = 1; i <= kSamples; ++i) {
      const double x = input.left + (step * static_cast<double>(i));
      best = std::min(best, input.objective(x));
    }
    return best;
  }

  InType input_{};
  double reference_min_ = 0.0;
};

namespace {

const std::array<TestType, 3> kTestParams = {
    TestType{-5.0, 5.0, 1e-3, 2.0, 250, [](double x) { return (x - 2.0) * (x - 2.0); }},
    TestType{-10.0, 10.0, 1e-3, 2.0, 300, [](double x) { return std::sin(x) + (0.1 * x); }},
    TestType{-3.0, 3.0, 1e-3, 2.0, 300, [](double x) { return std::fabs(x) + (0.2 * std::cos(10.0 * x)); }},
};

const auto kTaskList =
    std::tuple_cat(ppc::util::AddFuncTask<StronginSearchSeq, InType>(kTestParams, PPC_SETTINGS_global_search_strongin),
                   ppc::util::AddFuncTask<StronginSearchMpi, InType>(kTestParams, PPC_SETTINGS_global_search_strongin));

const auto kGTestValues = ppc::util::ExpandToValues(kTaskList);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(Strongin, StronginFuncTests, kGTestValues,
                         StronginFuncTests::PrintFuncTestName<StronginFuncTests>);

TEST_P(StronginFuncTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace global_search_strongin
