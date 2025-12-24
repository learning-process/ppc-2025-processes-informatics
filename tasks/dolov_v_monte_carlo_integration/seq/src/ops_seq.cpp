#include "dolov_v_monte_carlo_integration/seq/include/ops_seq.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "dolov_v_monte_carlo_integration/common/include/common.hpp"
#include "util/include/util.hpp"

namespace dolov_v_monte_carlo_integration {

DolovVMonteCarloIntegrationSEQ::DolovVMonteCarloIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool DolovVMonteCarloIntegrationSEQ::ValidationImpl() {  // NOLINT
  const auto &in = GetInput();
  // Явная проверка на nullptr для func и проверка остальных полей
  return static_cast<bool>(in.func) && (in.samples_count > 0) && (in.dimension > 0) &&
         (in.center.size() == static_cast<size_t>(in.dimension)) && (in.radius > 0.0);
}

bool DolovVMonteCarloIntegrationSEQ::PreProcessingImpl() {  // NOLINT
  GetOutput() = 0.0;
  return true;
}

bool DolovVMonteCarloIntegrationSEQ::RunImpl() {  // NOLINT
  const auto &in = GetInput();
  const int n_samples = in.samples_count;
  const int dim_val = in.dimension;
  const double rad = in.radius;
  const double r_sq = rad * rad;

  std::mt19937 generator(42);
  std::uniform_real_distribution<double> distribution(-rad, rad);

  double sum_val = 0.0;
  std::vector<double> point(static_cast<size_t>(dim_val));

  for (int sample = 0; sample < n_samples; ++sample) {
    for (int idx = 0; idx < dim_val; ++idx) {
      point[idx] = in.center[idx] + distribution(generator);
    }

    bool is_in_domain = true;
    if (in.domain_type == IntegrationDomain::kHyperSphere) {
      double distance_sq = 0.0;
      for (int idx = 0; idx < dim_val; ++idx) {
        double diff = point[idx] - in.center[idx];
        distance_sq += diff * diff;
      }

      if (distance_sq > r_sq) {
        is_in_domain = false;
      }
    }

    if (is_in_domain) {
      sum_val += in.func(point);
    }
  }

  const double v_cube = std::pow(2.0 * rad, dim_val);
  GetOutput() = v_cube * (sum_val / n_samples);

  return std::isfinite(GetOutput());
}

bool DolovVMonteCarloIntegrationSEQ::PostProcessingImpl() {  // NOLINT
  return true;
}

}  // namespace dolov_v_monte_carlo_integration
