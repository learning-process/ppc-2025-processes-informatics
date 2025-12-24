#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "dolov_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dolov_v_monte_carlo_integration {
namespace {

using InType = InputParams;
using OutType = double;
using TestType = std::tuple<int, std::string>;

double FuncSquareSum(const std::vector<double> &x) {
  double sum_val = 0.0;
  for (double val : x) {
    sum_val += val * val;
  }
  return sum_val;
}

class MonteCarloIntegrationTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    double tolerance = (std::get<0>(params) < 1000) ? 0.5 : 0.15;
    return std::abs(output_data - expected_result_) < tolerance;
  }
  InType GetTestInputData() final {
    return input_data_;
  }
  InType input_data_;
  OutType expected_result_ = 0.0;
};

class MonteCarloHyperCubeTests : public MonteCarloIntegrationTests {
 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);
    int dim = (std::get<1>(params) == "medium") ? 3 : 2;

    input_data_ = {.func = FuncSquareSum,
                   .dimension = dim,
                   .samples_count = samples,
                   .center = std::vector<double>(dim, 0.0),
                   .radius = 1.0,
                   .domain_type = IntegrationDomain::kHyperCube};
    expected_result_ = std::pow(2.0, dim) * (static_cast<double>(dim) / 3.0);
  }
};

class MonteCarloHyperSphereTests : public MonteCarloIntegrationTests {
 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);
    int dim = 2;
    double rad = 1.2;

    input_data_ = {.func = FuncSquareSum,
                   .dimension = dim,
                   .samples_count = samples,
                   .center = {0.0, 0.0},
                   .radius = rad,
                   .domain_type = IntegrationDomain::kHyperSphere};
    expected_result_ = (std::acos(-1.0) * std::pow(rad, 4)) / 2.0;
  }
};

TEST_P(MonteCarloHyperCubeTests, IntegrationHyperCube) {
  ExecuteTest(GetParam());
}
TEST_P(MonteCarloHyperSphereTests, IntegrationHyperSphere) {
  ExecuteTest(GetParam());
}

TEST(DolovVMonteCarloIntegrationFunctionalTest, ValidationFails) {
  InType in = {FuncSquareSum, -1, 0, {}, -1.0, IntegrationDomain::kHyperCube};
  DolovVMonteCarloIntegrationSEQ task_seq(in);
}

const std::array<TestType, 4> kTestParamsArray = {std::make_tuple(10000, "small"), std::make_tuple(50000, "medium"),
                                                  std::make_tuple(100000, "large"), std::make_tuple(500, "tiny")};

const auto kCubeTasksList = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                               kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                           ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                               kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration));
INSTANTIATE_TEST_SUITE_P(MonteCarloHyperCubeTests, MonteCarloHyperCubeTests, ppc::util::ExpandToValues(kCubeTasksList),
                         MonteCarloHyperCubeTests::PrintFuncTestName<MonteCarloHyperCubeTests>);

const auto kSphereTasksList = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                                 kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                             ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                                 kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration));
INSTANTIATE_TEST_SUITE_P(MonteCarloHyperSphereTests, MonteCarloHyperSphereTests,
                         ppc::util::ExpandToValues(kSphereTasksList),
                         MonteCarloHyperSphereTests::PrintFuncTestName<MonteCarloHyperSphereTests>);

}  // namespace
}  // namespace dolov_v_monte_carlo_integration
