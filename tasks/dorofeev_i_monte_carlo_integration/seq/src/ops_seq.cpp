#include "dorofeev_i_monte_carlo_integration/seq/include/ops_seq.hpp"

#include <cstddef>
#include <random>
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

  if (in.a.size() != in.b.size()) {
    return false;
  }

  if (in.samples <= 0) {
    return false;
  }

  for (size_t i = 0; i < in.a.size(); i++) {
    if (in.b[i] <= in.a[i]) {
      return false;
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
  const int dims = in.a.size();
  const int n = in.samples;

  std::mt19937 gen(12345);
  std::vector<std::uniform_real_distribution<double>> dist;
  dist.reserve(dims);

  for (int i = 0; i < dims; i++) {
    dist.emplace_back(in.a[i], in.b[i]);
  }

  double sum = 0.0;
  std::vector<double> point(dims);

  for (int s = 0; s < n; s++) {
    for (int d = 0; d < dims; d++) {
      point[d] = dist[d](gen);
    }
    sum += in.func(point);
  }

  double volume = 1.0;
  for (int i = 0; i < dims; i++) {
    volume *= (in.b[i] - in.a[i]);
  }

  GetOutput() = volume * (sum / n);
  return true;
}

bool DorofeevIMonteCarloIntegrationSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_monte_carlo_integration_processes
