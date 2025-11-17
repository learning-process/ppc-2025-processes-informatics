#include "dorofeev_i_monte_carlo_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <random>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"
#include "util/include/util.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {

DorofeevIMonteCarloIntegrationMPI::DorofeevIMonteCarloIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DorofeevIMonteCarloIntegrationMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (!in.func) {
    return false;
  }
  if (in.a.size() == 0 || in.a.size() != in.b.size()) {
    return false;
  }

  for (size_t i = 0; i < in.a.size(); i++) {
    if (in.b[i] <= in.a[i]) {
      return false;
    }
  }

  if (in.samples <= 0) {
    return false;
  }

  return true;
}

bool DorofeevIMonteCarloIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DorofeevIMonteCarloIntegrationMPI::RunImpl() {
  const auto in = GetInput();
  const int dims = in.a.size();
  const int N_total = in.samples;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N_local = N_total / size;
  if (rank == size - 1) {
    N_local += N_total % size;
  }

  std::vector<std::uniform_real_distribution<double>> dist;
  dist.reserve(dims);
  for (int d = 0; d < dims; ++d) {
    dist.emplace_back(in.a[d], in.b[d]);
  }

  std::mt19937 gen(rank + 123);
  double local_sum = 0.0;

  for (int i = 0; i < N_local; ++i) {
    std::vector<double> x(dims);
    for (int d = 0; d < dims; d++) {
      x[d] = dist[d](gen);
    }
    local_sum += in.func(x);
  }

  double global_sum = 0.0;

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double volume = 1.0;
  for (int d = 0; d < dims; d++) {
    volume *= (in.b[d] - in.a[d]);
  }

  double result = 0.0;
  if (rank == 0) {
    result = (global_sum / N_total) * volume;
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = result;
  return true;
}

bool DorofeevIMonteCarloIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_monte_carlo_integration_processes
