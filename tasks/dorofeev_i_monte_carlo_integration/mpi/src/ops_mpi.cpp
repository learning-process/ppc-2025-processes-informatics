#include "dorofeev_i_monte_carlo_integration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "dorofeev_i_monte_carlo_integration/common/include/common.hpp"

namespace dorofeev_i_monte_carlo_integration_processes {

DorofeevIMonteCarloIntegrationMPI::DorofeevIMonteCarloIntegrationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool DorofeevIMonteCarloIntegrationMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool valid = true;

  if (rank == 0) {
    const auto &in = GetInput();

    if (!in.func) {
      valid = false;
    }
    if (in.a.empty() || in.a.size() != in.b.size()) {
      valid = false;
    }

    for (size_t i = 0; i < in.a.size(); ++i) {
      if (in.b[i] <= in.a[i]) {
        valid = false;
      }
    }

    if (in.samples <= 0) {
      valid = false;
    }
  }

  MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return valid;
}

bool DorofeevIMonteCarloIntegrationMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool DorofeevIMonteCarloIntegrationMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // BROADCAST INPUT
  InType in;

  if (rank == 0) {
    in = GetInput();
  }

  // broadcast dims
  int dims = static_cast<int>(rank == 0 ? in.a.size() : 0);  // Because MPI doesn't support size_t
  MPI_Bcast(&dims, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // resize on other ranks
  if (rank != 0) {
    in.a.resize(dims);
    in.b.resize(dims);
  }

  // broadcast vectors a and b
  MPI_Bcast(in.a.data(), dims, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(in.b.data(), dims, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // broadcast samples
  MPI_Bcast(&in.samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // broadcast func id (always 1 for now)
  int func_id = (rank == 0 ? 1 : 0);
  MPI_Bcast(&func_id, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // restore the function
  switch (func_id) {
    case 1:
      in.func = [](const std::vector<double> &x) { return x[0] * x[0]; };
      break;
    default:
      in.func = nullptr;
  }

  // CALCULATIONS
  int n_total = in.samples;
  int n_local = n_total / size;
  if (rank == size - 1) {
    n_local += n_total % size;
  }

  std::vector<std::uniform_real_distribution<double>> dist;
  dist.reserve(dims);
  for (int dim = 0; std::cmp_less(dim, dims); dim++) {
    dist.emplace_back(in.a[dim], in.b[dim]);
  }

  std::mt19937 gen(rank + 777);
  double local_sum = 0.0;

  for (int i = 0; i < n_local; ++i) {
    std::vector<double> x(dims);
    for (int dim = 0; std::cmp_less(dim, dims); dim++) {
      x[dim] = dist[dim](gen);
    }
    local_sum += in.func(x);
  }

  // REDUCE
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // RESULT
  double result = 0.0;
  if (rank == 0) {
    double volume = 1.0;
    for (int dim = 0; std::cmp_less(dim, dims); dim++) {
      volume *= (in.b[dim] - in.a[dim]);
    }
    result = (global_sum / n_total) * volume;
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = result;
  return true;
}

bool DorofeevIMonteCarloIntegrationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_monte_carlo_integration_processes
