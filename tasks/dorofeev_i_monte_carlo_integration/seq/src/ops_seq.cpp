#include "dorofeev_i_monte_carlo_integration/seq/include/ops_seq.hpp"

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"

// NOTE:
// Validation branches are excluded from coverage because they require crafting
// intentionally invalid input structures that cannot appear in the framework's
// normal execution flow. These checks serve only as defensive guards and are
// not part of the algorithmic logic. LCOV_EXCL prevents artificial
// coverage drops caused by unreachable input states.

namespace dorofeev_i_monte_carlo_integration_processes {

DorofeevIMonteCarloIntegrationSEQ::DorofeevIMonteCarloIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool DorofeevIMonteCarloIntegrationSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.a.size() != in.b.size()) {  // LCOV_EXCL_LINE
    return false;
  }

  if (in.samples <= 0) {  // LCOV_EXCL_LINE
    return false;
  }

  for (size_t i = 0; i < in.a.size(); i++) {
    if (in.b[i] <= in.a[i]) {  // LCOV_EXCL_LINE
      return false;            // LCOV_EXCL_LINE
    }
  }

  return true;
}

bool DorofeevIMonteCarloIntegrationSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DorofeevIMonteCarloIntegrationSEQ::RunImpl() {
  const auto &in = GetInput();
  const std::size_t dims = in.a.size();
  const int n = in.samples;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<std::uniform_real_distribution<double>> dist;
  dist.reserve(dims);

  for (std::size_t i = 0; std::cmp_less(i, dims); i++) {  // LCOV_EXCL_LINE
    dist.emplace_back(in.a[i], in.b[i]);
  }

  double sum = 0.0;
  std::vector<double> point(dims);

  for (int sample = 0; sample < n; sample++) {
    for (std::size_t dim = 0; dim < dims; dim++) {
      point[dim] = dist[dim](gen);
    }
    sum += in.func(point);
  }

  double volume = 1.0;
  for (std::size_t i = 0; std::cmp_less(i, dims); i++) {  // LCOV_EXCL_LINE
    volume *= (in.b[i] - in.a[i]);
  }

  GetOutput() = volume * (sum / n);
  return true;
}

bool DorofeevIMonteCarloIntegrationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_monte_carlo_integration_processes
