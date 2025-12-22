#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "dolov_v_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dolov_v_monte_carlo_integration {
namespace {

class DolovVMonteCarloIntegrationPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  // Количество семплов для Performance-теста
  static constexpr int kSamples = 1000000;

  InType input_data_{};
  OutType expected_result_ = 0.0;

  // Функция для тестирования: f(x1, x2, x3) = x1 + x2 + x3, (Точный интеграл на [-1, 1]^3 равен 0)
  static double func_sum_coords_3d(const std::vector<double> &x) {
    double sum = 0.0;
    for (double val : x) {
      sum += val;
    }
    return sum;
  }

 protected:
  void SetUp() override {
    //  Инициализация InType для 3D интегрирования
    const int kDimension = 3;
    const double kRadius = 1.0;
    const std::vector<double> kCenter = {0.0, 0.0, 0.0};

    input_data_ = {.func = func_sum_coords_3d,
                   .dimension = kDimension,
                   .samples_count = kSamples,
                   .center = kCenter,
                   .radius = kRadius,
                   .domain_type = IntegrationDomain::kHyperCube};

    // Интеграл f(x1 + x2 + x3) на симметричном кубе [-1, 1]^3 равен 0.
    expected_result_ = 0.0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double kTolerance = 0.02;
    return std::abs(output_data - expected_result_) < kTolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DolovVMonteCarloIntegrationPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DolovVMonteCarloIntegrationMPI, DolovVMonteCarloIntegrationSEQ>(
        PPC_SETTINGS_dolov_v_monte_carlo_integration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DolovVMonteCarloIntegrationPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(MonteCarlo3DPerf, DolovVMonteCarloIntegrationPerfTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace dolov_v_monte_carlo_integration
