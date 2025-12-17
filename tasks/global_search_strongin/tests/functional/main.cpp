#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "global_search_strongin/common/include/common.hpp"
#include "global_search_strongin/mpi/include/ops_mpi.hpp"
#include "global_search_strongin/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace global_search_strongin {

struct StronginCase {
  double left;
  double right;
  double reliability;
  double epsilon;
  int max_iterations;
  std::function<double(double)> function;
  double sample_step;
};

using TestType = StronginCase;

namespace {

OutType SampleReference(const StronginCase &test_case) {
  OutType result{};
  result.best_point = test_case.left;
  result.best_value = test_case.function(test_case.left);
  for (double x = test_case.left; x <= test_case.right; x += test_case.sample_step) {
    const double value = test_case.function(x);
    if (value < result.best_value) {
      result.best_value = value;
      result.best_point = x;
    }
  }
  const double value_right = test_case.function(test_case.right);
  if (value_right < result.best_value) {
    result.best_value = value_right;
    result.best_point = test_case.right;
  }
  return result;
}

}  // namespace

class StronginFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    const std::string left_tag =
        std::string(param.left < 0.0 ? "M" : "P") + std::to_string(static_cast<int>(std::abs(param.left)));
    const std::string right_tag =
        std::string(param.right < 0.0 ? "M" : "P") + std::to_string(static_cast<int>(std::abs(param.right)));
    return left_tag + "_" + right_tag;
  }

 protected:
  void SetUp() override {
    current_case_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_.left = current_case_.left;
    input_.right = current_case_.right;
    input_.reliability = current_case_.reliability;
    input_.epsilon = current_case_.epsilon;
    input_.max_iterations = current_case_.max_iterations;
    input_.objective = current_case_.function;

    reference_ = SampleReference(current_case_);
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double eps = std::max(current_case_.epsilon * 2.0, current_case_.sample_step * 2.0);
    const bool point_ok = std::fabs(output_data.best_point - reference_.best_point) <= eps;
    const bool value_ok = std::fabs(output_data.best_value - reference_.best_value) <= eps;
    return point_ok && value_ok;
  }

 private:
  InType input_{};
  OutType reference_{};
  TestType current_case_{};
};

namespace {

const std::array<TestType, 3> kTestCases = {
    TestType{0.0, 3.0, 2.0, 1e-3, 200, [](double x) { return (x - 1.3) * (x - 1.3); }, 1e-4},
    TestType{-2.0, 2.0, 3.0, 5e-3, 250, [](double x) { return std::sin(x) + 0.2 * x; }, 1e-3},
    TestType{0.0, 6.0, 2.5, 5e-3, 300, [](double x) {
  if (x < 2.0) {
    return std::cos(x) + 0.1 * x;
  }
  return std::sin(1.5 * x) + 0.05 * (x - 2.0);
}, 1e-3}};

const auto kTaskList =
    std::tuple_cat(ppc::util::AddFuncTask<StronginSearchSeq, InType>(kTestCases, PPC_SETTINGS_global_search_strongin),
                   ppc::util::AddFuncTask<StronginSearchMpi, InType>(kTestCases, PPC_SETTINGS_global_search_strongin));

const auto kGTestValues = ppc::util::ExpandToValues(kTaskList);

INSTANTIATE_TEST_SUITE_P(StronginSearch, StronginFuncTests, kGTestValues,
                         StronginFuncTests::PrintFuncTestName<StronginFuncTests>);

TEST_P(StronginFuncTests, Runs) {
  ExecuteTest(GetParam());
}

}  // namespace

}  // namespace global_search_strongin
