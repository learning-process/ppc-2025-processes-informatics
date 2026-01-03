#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "global_search_strongin/common/include/common.hpp"
#include "global_search_strongin/mpi/include/ops_mpi.hpp"
#include "global_search_strongin/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace global_search_strongin {

namespace {

double ComputeReferenceMin(const InType &input) {
  constexpr std::size_t kSamples = 50000;
  const double step = (input.right - input.left) / static_cast<double>(kSamples);
  double best = input.objective(input.left);
  for (std::size_t i = 1; i <= kSamples; ++i) {
    const double x = input.left + (step * static_cast<double>(i));
    best = std::min(best, input.objective(x));
  }
  return best;
}

}  // namespace

class StronginPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_.left = -10.0;
    input_.right = 10.0;
    input_.epsilon = 1e-3;
    input_.reliability = 2.0;
    input_.max_iterations = 500;
    input_.objective = [](double x) {
      double acc = 0.0;
      for (int i = 1; i <= 3000; ++i) {
        const double t = x * static_cast<double>(i);
        acc += std::sin(t) * std::cos(t / (static_cast<double>(i) + 1.0));
      }

      const double base = std::sin(x) + (0.25 * std::sin(3.0 * x)) + (0.01 * x);
      volatile double sink = acc;
      return base + (static_cast<double>(sink) * 1e-12);
    };
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
    return output_data.best_value <= reference_min_ + 5e-2;
  }

 private:
  InType input_{};
  double reference_min_ = 0.0;
};

namespace {

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, StronginSearchMpi, StronginSearchSeq>(PPC_SETTINGS_global_search_strongin);

const auto kGTestValues = ppc::util::TupleToGTestValues(kPerfTasks);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(StronginPerf, StronginPerfTests, kGTestValues, StronginPerfTests::CustomPerfTestName);

TEST_P(StronginPerfTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace global_search_strongin
