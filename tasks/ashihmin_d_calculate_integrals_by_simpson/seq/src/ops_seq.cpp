#include "ashihmin_d_calculate_integrals_by_simpson/seq/include/ops_seq.hpp"

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

AshihminDCalculateIntegralsBySimpsonSEQ::AshihminDCalculateIntegralsBySimpsonSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0.0;
}

bool AshihminDCalculateIntegralsBySimpsonSEQ::ValidationImpl() {
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

bool AshihminDCalculateIntegralsBySimpsonSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

double AshihminDCalculateIntegralsBySimpsonSEQ::IntegrandFunction(const std::vector<double> &coordinates) {
  return Function(coordinates);
}

void AshihminDCalculateIntegralsBySimpsonSEQ::RecursiveIntegration(int dimension, std::vector<double> &coordinates,
                                                                   const std::vector<double> &step_sizes,
                                                                   double &total_sum) {
  const auto &input = GetInput();
  int partitions = input.partitions;
  int total_dimensions = static_cast<int>(coordinates.size());

  if (dimension == total_dimensions) {
    double weight_coefficient = 1.0;
    for (const double coordinate : coordinates) {
      int node_index = static_cast<int>(coordinate);

      if (node_index == 0 || node_index == partitions) {
        weight_coefficient *= 1.0;
      } else if (node_index % 2 == 0) {
        weight_coefficient *= 2.0;
      } else {
        weight_coefficient *= 4.0;
      }
    }

    std::vector<double> point(total_dimensions);
    for (int dim_index = 0; dim_index < total_dimensions; ++dim_index) {
      point[dim_index] = input.left_bounds[dim_index] + (coordinates[dim_index] * step_sizes[dim_index]);
    }

    total_sum += weight_coefficient * IntegrandFunction(point);
    return;
  }

  for (int node_index = 0; node_index <= partitions; ++node_index) {
    coordinates[dimension] = static_cast<double>(node_index);
    RecursiveIntegration(dimension + 1, coordinates, step_sizes, total_sum);
  }
}

bool AshihminDCalculateIntegralsBySimpsonSEQ::RunImpl() {
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

  double total_sum = 0.0;

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

    double weight_coefficient = 1.0;
    for (int dim = 0; dim < dimensions; ++dim) {
      int node_index = indices[static_cast<std::size_t>(dim)];
      if (node_index == 0 || node_index == partitions) {
        weight_coefficient *= 1.0;
      } else if (node_index % 2 == 0) {
        weight_coefficient *= 2.0;
      } else {
        weight_coefficient *= 4.0;
      }
    }

    std::vector<double> point(static_cast<std::size_t>(dimensions));
    for (int dim = 0; dim < dimensions; ++dim) {
      point[static_cast<std::size_t>(dim)] =
          input.left_bounds[static_cast<std::size_t>(dim)] +
          indices[static_cast<std::size_t>(dim)] * step_sizes[static_cast<std::size_t>(dim)];
    }

    total_sum += weight_coefficient * IntegrandFunction(point);
  }

  double volume_element = 1.0;
  for (int dim_index = 0; dim_index < dimensions; ++dim_index) {
    volume_element *= step_sizes[static_cast<std::size_t>(dim_index)];
  }

  GetOutput() = total_sum * volume_element / std::pow(3.0, dimensions);
  return true;
}

bool AshihminDCalculateIntegralsBySimpsonSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace ashihmin_d_calculate_integrals_by_simpson
