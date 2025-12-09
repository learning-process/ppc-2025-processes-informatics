#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "dolov_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dolov_v_monte_carlo_integration {

using InType = InputParams;
using OutType = double;
using TestType = std::tuple<int, std::string>;

// Вспомогательная функция для тестовой функции: f(x1, x2) = x1^2 + x2^2
inline double func_square_sum(const std::vector<double> &x) {
  double sum = 0.0;
  for (double val : x) {
    sum += val * val;
  }
  return sum;
}

// Класс тестирования: гиперкуб

class MonteCarloHyperCubeTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);

    // Параметры для интегрирования в 2D Кубе
    const int kDimension = 2;
    const double kRadius = 1.0;
    const std::vector<double> kCenter = {0.0, 0.0};

    input_data_ = {.func = func_square_sum,
                   .dimension = kDimension,
                   .samples_count = samples,
                   .center = kCenter,
                   .radius = kRadius,
                   .domain_type = IntegrationDomain::kHyperCube};

    // Точный результат для Int[(x1^2 + x2^2) dx1 dx2] на [-1, 1]^2 равен 8/3.
    expected_result_ = 8.0 / 3.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double kTolerance = 0.05;
    return std::abs(output_data - expected_result_) < kTolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0.0;
};

// Класс тестирование: гиперсфера
class MonteCarloHyperSphereTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int samples = std::get<0>(params);

    // Параметры для интегрирования в 2D Сфере радиусом 1 (центр 0,0)
    const int kDimension = 2;
    const double kRadius = 1.0;
    const std::vector<double> kCenter = {0.0, 0.0};

    input_data_ = {.func = func_square_sum,
                   .dimension = kDimension,
                   .samples_count = samples,
                   .center = kCenter,
                   .radius = kRadius,
                   .domain_type = IntegrationDomain::kHyperSphere};

    // Точный результат для Int[(x1^2 + x2^2) dx1 dx2] на единичной 2D сфере равен Pi/2.
    expected_result_ = M_PI / 2.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double kTolerance = 0.05;
    return std::abs(output_data - expected_result_) < kTolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0.0;
};

namespace {

// Тестовые вызовы
//  Функциональный вызов для КУБА
TEST_P(MonteCarloHyperCubeTests, IntegrationHyperCube2D) {
  ExecuteTest(GetParam());
}

// Функциональный вызов для СФЕРЫ
TEST_P(MonteCarloHyperSphereTests, IntegrationHyperSphere2D) {
  ExecuteTest(GetParam());
}

// Параметры для тестирования (Samples Count, Name)
const std::array<TestType, 3> kTestParam = {std::make_tuple(10000, "small"), std::make_tuple(50000, "medium"),
                                            std::make_tuple(200000, "large")};

// 1. Создание списка задач для куба
const auto kCubeTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                                   kTestParam, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                               ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                                   kTestParam, PPC_SETTINGS_dolov_v_monte_carlo_integration));

const auto kCubeGtestValues = ppc::util::ExpandToValues(kCubeTestTasksList);

const auto kCubeTestName = MonteCarloHyperCubeTests::PrintFuncTestName<MonteCarloHyperCubeTests>;

INSTANTIATE_TEST_SUITE_P(MonteCarloHyperCubeTests, MonteCarloHyperCubeTests, kCubeGtestValues, kCubeTestName);

// 2. Создание списка задач для сферы
const auto kSphereTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<DolovVMonteCarloIntegrationMPI, InType>(
                                                     kTestParam, PPC_SETTINGS_dolov_v_monte_carlo_integration),
                                                 ppc::util::AddFuncTask<DolovVMonteCarloIntegrationSEQ, InType>(
                                                     kTestParam, PPC_SETTINGS_dolov_v_monte_carlo_integration));

const auto kSphereGtestValues = ppc::util::ExpandToValues(kSphereTestTasksList);

const auto kSphereTestName = MonteCarloHyperSphereTests::PrintFuncTestName<MonteCarloHyperSphereTests>;

INSTANTIATE_TEST_SUITE_P(MonteCarloHyperSphereTests, MonteCarloHyperSphereTests, kSphereGtestValues, kSphereTestName);

}  // namespace
}  // namespace dolov_v_monte_carlo_integration
