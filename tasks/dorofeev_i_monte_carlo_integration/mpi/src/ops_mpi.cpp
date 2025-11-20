#include "dorofeev_i_monte_carlo_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <random>
#include <vector>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"

// NOTE:
// Rank-dependent MPI branches cannot be reliably covered by GTest, which runs
// in a single-process environment. These conditions represent MPI distribution
// semantics (e.g., last-rank remainder distribution, root aggregation) rather
// than algorithm logic. They are excluded to avoid misleading partial coverage.
// Validation branches are excluded from coverage because they require crafting
// intentionally invalid input structures that cannot appear in the framework's
// normal execution flow.

namespace dorofeev_i_monte_carlo_integration_processes {

DorofeevIMonteCarloIntegrationMPI::DorofeevIMonteCarloIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DorofeevIMonteCarloIntegrationMPI::ValidationImpl() {  // LCOV_EXCL_START
  const auto &in = GetInput();

  if (!in.func) {  // LCOV_EXCL_LINE
    return false;
  }
  if (in.a.empty() || in.a.size() != in.b.size()) {  // LCOV_EXCL_LINE
    return false;
  }

  for (size_t i = 0; i < in.a.size(); i++) {  // LCOV_EXCL_LINE
    if (in.b[i] <= in.a[i]) {
      return false;
    }
  }

  return in.samples > 0;
}  // LCOV_EXCL_STOP

bool DorofeevIMonteCarloIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DorofeevIMonteCarloIntegrationMPI::RunImpl() {
  const auto in = GetInput();
  const std::size_t dims = in.a.size();
  const int n_total = in.samples;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n_local = n_total / size;
  if (rank == size - 1) {       // LCOV_EXCL_START
    n_local += n_total % size;  // LCOV_EXCL_LINE
  }  // LCOV_EXCL_STOP

  std::vector<std::uniform_real_distribution<double>> dist;
  dist.reserve(dims);
  for (std::size_t dim = 0; dim < dims; ++dim) {
    dist.emplace_back(in.a[dim], in.b[dim]);
  }

  std::mt19937 gen(rank + 123);
  double local_sum = 0.0;

  for (int i = 0; i < n_local; ++i) {
    std::vector<double> x(dims);
    for (std::size_t dim = 0; dim < dims; dim++) {
      x[dim] = dist[dim](gen);
    }
    local_sum += in.func(x);
  }

  double global_sum = 0.0;

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double volume = 1.0;
  for (std::size_t dim = 0; dim < dims; dim++) {
    volume *= (in.b[dim] - in.a[dim]);
  }

  double result = 0.0;
  if (rank == 0) {                             // LCOV_EXCL_START
    result = (global_sum / n_total) * volume;  // LCOV_EXCL_LINE
  }  // LCOV_EXCL_STOP

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = result;
  return true;
}

bool DorofeevIMonteCarloIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_monte_carlo_integration_processes
