#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "eremin_v_rectangle_method/common/include/common.hpp"
#include "eremin_v_rectangle_method/mpi/include/ops_mpi.hpp"
#include "eremin_v_rectangle_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace eremin_v_rectangle_method {

class EreminVRunFuncTestsRectangleMethod : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string result = "test_" + std::to_string(std::get<0>(test_param)) + "_from_" +
                         std::to_string(std::get<1>(test_param)) + "_to_" + std::to_string(std::get<2>(test_param)) +
                         "_steps_" + std::to_string(std::get<3>(test_param));
    std::ranges::replace(result, '.', '_');
    std::ranges::replace(result, '-', 'm');
    return result;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    double lower_bound = std::get<1>(params);
    double upper_bound = std::get<2>(params);
    int steps = std::get<3>(params);
    input_data_ = std::make_tuple(lower_bound, upper_bound, steps, InFunction);

    expected_result_ = Function(upper_bound) - Function(lower_bound);
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
    return (std::exp(x) * std::sin(x)) + (x * x * x * std::cos(x));
  }
  static double InFunction(double x) {
    return (std::exp(x) * (std::sin(x) + std::cos(x))) + (3 * x * x * std::cos(x)) - (x * x * x * std::sin(x));
  }
};

namespace {

TEST_P(EreminVRunFuncTestsRectangleMethod, RectangleMethod) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(1, 0.0, 1.0, 1000), std::make_tuple(2, 0.0, 2.0, 10000),
                                            std::make_tuple(3, -10.0, 100.0, 100000),
                                            std::make_tuple(4, 1.0, 4.0, 50000), std::make_tuple(5, -2.0, 3.0, 80000)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<EreminVRectangleMethodMPI, InType>(kTestParam, PPC_SETTINGS_eremin_v_rectangle_method),
    ppc::util::AddFuncTask<EreminVRectangleMethodSEQ, InType>(kTestParam, PPC_SETTINGS_eremin_v_rectangle_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = EreminVRunFuncTestsRectangleMethod::PrintFuncTestName<EreminVRunFuncTestsRectangleMethod>;

INSTANTIATE_TEST_SUITE_P(RectangleMethodTests, EreminVRunFuncTestsRectangleMethod, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace eremin_v_rectangle_method
