#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "global_search_strongin/common/include/common.hpp"
#include "global_search_strongin/mpi/include/ops_mpi.hpp"
#include "global_search_strongin/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace global_search_strongin {

namespace {

OutType SampleReference(const InType &input, double step) {
  OutType result{};
  result.best_point = input.left;
  result.best_value = input.objective(input.left);
  const double span = input.right - input.left;
  const int steps = span <= 0.0 ? 0 : static_cast<int>(std::ceil(span / step));
  for (int sample_index = 1; sample_index <= steps; ++sample_index) {
    const double current_point = std::min(input.right, input.left + (static_cast<double>(sample_index) * step));
    const double value = input.objective(current_point);
    if (value < result.best_value) {
      result.best_value = value;
      result.best_point = current_point;
    }
  }
  return result;
}

}  // namespace

class StronginPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_.left = -2.0;
    input_.right = 5.0;
    input_.reliability = 2.5;
    input_.epsilon = 1e-3;
    input_.max_iterations = 600;
    input_.objective = [](double x_value) {
      return std::sin(x_value) + (std::cos(2.0 * x_value) * 0.25) + (0.02 * (x_value - 1.0) * (x_value - 1.0));
    };
    reference_ = SampleReference(input_, 5e-4);
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = 1e-2;
    const bool point_ok = std::fabs(output_data.best_point - reference_.best_point) <= eps;
    const bool value_ok = std::fabs(output_data.best_value - reference_.best_value) <= eps;
    return point_ok && value_ok;
  }

 private:
  InType input_{};
  OutType reference_{};
};

namespace {

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, StronginSearchMpi, StronginSearchSeq>(PPC_SETTINGS_global_search_strongin);

const auto kGTestValues = ppc::util::TupleToGTestValues(kPerfTasks);

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)
INSTANTIATE_TEST_SUITE_P(StronginPerf, StronginPerfTests, kGTestValues, StronginPerfTests::CustomPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables, modernize-type-traits)

TEST_P(StronginPerfTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace global_search_strongin
