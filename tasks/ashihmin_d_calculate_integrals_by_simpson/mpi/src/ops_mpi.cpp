#include "ashihmin_d_calculate_integrals_by_simpson/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ashihmin_d_calculate_integrals_by_simpson {

namespace {
double Function(const std::vector<double> &coordinates) {
  double total = 0.0;
  for (double value : coordinates) {
    total += value * value;
  }
  return total;
}
}  // namespace

AshihminDCalculateIntegralsBySimpsonMPI::AshihminDCalculateIntegralsBySimpsonMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0.0;
}

bool AshihminDCalculateIntegralsBySimpsonMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.left_bounds.size() != input.right_bounds.size()) {
    return false;
  }
  if (input.partitions <= 0 || input.partitions % 2 != 0) {
    return false;
  }

  for (std::size_t index = 0; index < input.left_bounds.size(); ++index) {
    if (input.right_bounds[index] <= input.left_bounds[index]) {
      return false;
    }
  }
  return true;
}

bool AshihminDCalculateIntegralsBySimpsonMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

double AshihminDCalculateIntegralsBySimpsonMPI::IntegrandFunction(const std::vector<double> &coordinates) {
  return Function(coordinates);
}

static double CalculateWeightCoefficient(const std::vector<int> &indices, int partitions) {
  double weight_coefficient = 1.0;
  for (std::size_t dim = 0; dim < indices.size(); ++dim) {
    int node_index = indices[dim];
    if (node_index == 0 || node_index == partitions) {
      weight_coefficient *= 1.0;
    } else if (node_index % 2 == 0) {
      weight_coefficient *= 2.0;
    } else {
      weight_coefficient *= 4.0;
    }
  }
  return weight_coefficient;
}

static std::vector<double> CalculatePoint(const std::vector<int> &indices, const std::vector<double> &step_sizes,
                                          const std::vector<double> &left_bounds) {
  std::vector<double> point(indices.size());
  for (std::size_t dim = 0; dim < indices.size(); ++dim) {
    point[dim] = left_bounds[dim] + (indices[dim] * step_sizes[dim]);
  }
  return point;
}

bool AshihminDCalculateIntegralsBySimpsonMPI::RunImpl() {
  int process_rank = 0;
  int process_count = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  const auto &input = GetInput();
  int dimensions = static_cast<int>(input.left_bounds.size());
  int partitions = input.partitions;

  if (dimensions == 0) {
    GetOutput() = 0.0;
    return true;
  }

  std::vector<double> step_sizes(static_cast<std::size_t>(dimensions));
  for (int dim_index = 0; dim_index < dimensions; ++dim_index) {
    step_sizes[static_cast<std::size_t>(dim_index)] = (input.right_bounds[static_cast<std::size_t>(dim_index)] -
                                                       input.left_bounds[static_cast<std::size_t>(dim_index)]) /
                                                      partitions;
  }

  int total_nodes_first_dim = partitions + 1;
  int chunk_size = total_nodes_first_dim / process_count;
  int remainder = total_nodes_first_dim % process_count;

  int start_index = (process_rank * chunk_size) + std::min(process_rank, remainder);
  int end_index = start_index + chunk_size - 1;
  if (process_rank < remainder) {
    end_index += 1;
  }

  double local_sum = 0.0;

  if (start_index <= end_index) {
    std::vector<int> indices(static_cast<std::size_t>(dimensions), 0);

    int total_points = 1;
    for (int dim = 0; dim < dimensions; ++dim) {
      total_points *= (partitions + 1);
    }

    for (int point_idx = 0; point_idx < total_points; ++point_idx) {
      int temp = point_idx;
      for (int dim = 0; dim < dimensions; ++dim) {
        indices[static_cast<std::size_t>(dim)] = temp % (partitions + 1);
        temp /= (partitions + 1);
      }

      if (indices[0] < start_index || indices[0] > end_index) {
        continue;
      }

      double weight_coefficient = CalculateWeightCoefficient(indices, partitions);
      std::vector<double> point = CalculatePoint(indices, step_sizes, input.left_bounds);
      local_sum += weight_coefficient * IntegrandFunction(point);
    }
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double volume_element = 1.0;
  for (int dim_index = 0; dim_index < dimensions; ++dim_index) {
    volume_element *= step_sizes[static_cast<std::size_t>(dim_index)];
  }

  double result = global_sum * volume_element / std::pow(3.0, dimensions);

  if (process_rank == 0) {
    GetOutput() = result;
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool AshihminDCalculateIntegralsBySimpsonMPI::PostProcessingImpl() {
  return true;
}
}  // namespace ashihmin_d_calculate_integrals_by_simpson
