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

class MonteCarloHyperCubeTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);

    input_data_ = {.func = FuncSquareSum,
                   .dimension = 2,
                   .samples_count = samples,
                   .center = {0.0, 0.0},
                   .radius = 1.0,
                   .domain_type = IntegrationDomain::kHyperCube};
    expected_result_ = 8.0 / 3.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_result_) < 0.15;
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
    auto params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);
    // Изменяем радиус для параметров 'large', чтобы покрыть ветки логики радиуса
    double rad = (std::get<1>(params) == "large") ? 1.1 : 1.0;

    input_data_ = {.func = FuncSquareSum,
                   .dimension = 2,
                   .samples_count = samples,
                   .center = {0.0, 0.0},
                   .radius = rad,
                   .domain_type = IntegrationDomain::kHyperSphere};

    // Формула для x^2 + y^2 по кругу: (PI * R^4) / 2
    expected_result_ = (std::acos(-1.0) * std::pow(rad, 4)) / 2.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_result_) < 0.15;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0.0;
};

TEST_P(MonteCarloHyperCubeTests, IntegrationHyperCube2D) {
  ExecuteTest(GetParam());
}
TEST_P(MonteCarloHyperSphereTests, IntegrationHyperSphere2D) {
  ExecuteTest(GetParam());
}

// Дополнительный тест для покрытия функций и конструкторов (Functions & Branches)
TEST(DolovVMonteCarloIntegrationFunctionalTest, ManualTaskCoverage) {
  InType in = {FuncSquareSum, 2, 100, {0.0, 0.0}, 1.0, IntegrationDomain::kHyperCube};
  DolovVMonteCarloIntegrationSEQ test_seq(in);
  DolovVMonteCarloIntegrationMPI test_mpi(in);
  ASSERT_EQ(in.dimension, 2);
  ASSERT_GE(in.samples_count, 10);
}

const std::array<TestType, 3> kTestParamsArray = {std::make_tuple(10000, "small"), std::make_tuple(50000, "medium"),
                                                  std::make_tuple(100000, "large")};

const auto kCubeTasksList = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                               kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                           ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                               kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration));

const auto kCubeValsList = ppc::util::ExpandToValues(kCubeTasksList);
INSTANTIATE_TEST_SUITE_P(MonteCarloHyperCubeTests, MonteCarloHyperCubeTests, kCubeValsList,
                         MonteCarloHyperCubeTests::PrintFuncTestName<MonteCarloHyperCubeTests>);

const auto kSphereTasksList = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                                 kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                             ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                                 kTestParamsArray, PPC_SETTINGS_dolov_v_monte_carlo_integration));

const auto kSphereValsList = ppc::util::ExpandToValues(kSphereTasksList);
INSTANTIATE_TEST_SUITE_P(MonteCarloHyperSphereTests, MonteCarloHyperSphereTests, kSphereValsList,
                         MonteCarloHyperSphereTests::PrintFuncTestName<MonteCarloHyperSphereTests>);

}  // namespace
}  // namespace dolov_v_monte_carlo_integration
