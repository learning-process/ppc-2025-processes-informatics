#include "dorofeev_i_monte_carlo_integration/seq/include/ops_seq.hpp"

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {

DorofeevIMonteCarloIntegrationSEQ::DorofeevIMonteCarloIntegrationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool DorofeevIMonteCarloIntegrationSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (!in.func) {
    return false;
  }
  if (in.a.empty() || in.a.size() != in.b.size()) {
    return false;
  }

  for (size_t i = 0; i < in.a.size(); ++i) {
    if (in.b[i] <= in.a[i]) {
      return false;
    }
  }

  return in.samples > 0;
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
