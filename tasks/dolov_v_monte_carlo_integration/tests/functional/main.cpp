#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "dolov_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace dolov_v_monte_carlo_integration {

using InType = InputParams;
using OutType = double;
using TestType = std::tuple<int, std::string>;

// Вспомогательная функция: f(x1, x2) = x1^2 + x2^2 (PascalCase для Clang-Tidy)
inline double FuncSquareSum(const std::vector<double> &x) {
  double sum_val = 0.0;
  for (double val : x) {
    sum_val += val * val;
  }
  return sum_val;
}

class MonteCarloHyperCubeTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);

    const int k_dim = 2;
    const double k_rad = 1.0;
    const std::vector<double> k_center = {0.0, 0.0};

    input_data_ = {.func = FuncSquareSum,
                   .dimension = k_dim,
                   .samples_count = samples,
                   .center = k_center,
                   .radius = k_rad,
                   .domain_type = IntegrationDomain::kHyperCube};

    expected_result_ = 8.0 / 3.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double k_tolerance = 0.05;
    return std::abs(output_data - expected_result_) < k_tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0.0;
};

class MonteCarloHyperSphereTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);

    const int k_dim = 2;
    const double k_rad = 1.0;
    const std::vector<double> k_center = {0.0, 0.0};

    input_data_ = {.func = FuncSquareSum,
                   .dimension = k_dim,
                   .samples_count = samples,
                   .center = k_center,
                   .radius = k_rad,
                   .domain_type = IntegrationDomain::kHyperSphere};

    expected_result_ = M_PI / 2.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double k_tolerance = 0.05;
    return std::abs(output_data - expected_result_) < k_tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0.0;
};

namespace {
TEST_P(MonteCarloHyperCubeTests, IntegrationHyperCube2D) {
  ExecuteTest(GetParam());
}
TEST_P(MonteCarloHyperSphereTests, IntegrationHyperSphere2D) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> k_test_p = {std::make_tuple(10000, "small"), std::make_tuple(50000, "medium"),
                                          std::make_tuple(200000, "large")};

const auto k_cube_tasks = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                             k_test_p, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                         ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                             k_test_p, PPC_SETTINGS_dolov_v_monte_carlo_integration));

const auto k_cube_vals = ppc::util::ExpandToValues(k_cube_tasks);
const auto k_cube_name = MonteCarloHyperCubeTests::PrintFuncTestName<MonteCarloHyperCubeTests>;
INSTANTIATE_TEST_SUITE_P(MonteCarloHyperCubeTests, MonteCarloHyperCubeTests, k_cube_vals, k_cube_name);

const auto k_sphere_tasks = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                               k_test_p, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                           ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                               k_test_p, PPC_SETTINGS_dolov_v_monte_carlo_integration));

const auto k_sphere_vals = ppc::util::ExpandToValues(k_sphere_tasks);
const auto k_sphere_name = MonteCarloHyperSphereTests::PrintFuncTestName<MonteCarloHyperSphereTests>;
INSTANTIATE_TEST_SUITE_P(MonteCarloHyperSphereTests, MonteCarloHyperSphereTests, k_sphere_vals, k_sphere_name);
}  // namespace
}  // namespace dolov_v_monte_carlo_integration
