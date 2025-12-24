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

bool DolovVMonteCarloIntegrationSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return static_cast<bool>(in.func) && (in.samples_count > 0) && (in.dimension > 0) &&
         (in.center.size() == static_cast<size_t>(in.dimension)) && (in.radius > 0.0);
}

bool DolovVMonteCarloIntegrationSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DolovVMonteCarloIntegrationSEQ::RunImpl() {
  const auto &in = GetInput();
  const int n_val = in.samples_count;
  const int d_val = in.dimension;
  const double r_val = in.radius;
  const double r_sq = r_val * r_val;

  std::mt19937 generator(42);
  std::uniform_real_distribution<double> distribution(-r_val, r_val);

  double sum_res = 0.0;
  std::vector<double> point(static_cast<size_t>(d_val));

  for (int i = 0; i < n_val; ++i) {
    for (int j = 0; j < d_val; ++j) {
      point[j] = in.center[j] + distribution(generator);
    }

    bool inside = true;
    if (in.domain_type == IntegrationDomain::kHyperSphere) {
      double d2 = 0.0;
      for (int j = 0; j < d_val; ++j) {
        double diff = point[j] - in.center[j];
        d2 += diff * diff;
      }
      if (d2 > r_sq) {
        inside = false;
      }
    }

    if (inside) {
      sum_res += in.func(point);
    }
  }

  const double v_cube = std::pow(2.0 * r_val, d_val);
  GetOutput() = v_cube * (sum_res / n_val);

  return std::isfinite(GetOutput());
}

bool DolovVMonteCarloIntegrationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dolov_v_monte_carlo_integration
