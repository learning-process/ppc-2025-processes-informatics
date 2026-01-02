#include "ashihmin_d_calculate_integrals_by_simpson/seq/include/ops_seq.hpp"

#include <cmath>
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

  for (size_t index = 0; index < input.left_bounds.size(); ++index) {
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

  if (dimension == static_cast<int>(coordinates.size())) {
    double weight_coefficient = 1.0;
    for (int dim_index = 0; dim_index < static_cast<int>(coordinates.size()); ++dim_index) {
      int node_index = static_cast<int>(coordinates[dim_index]);

      if (node_index == 0 || node_index == partitions) {
        weight_coefficient *= 1.0;
      } else if (node_index % 2 == 0) {
        weight_coefficient *= 2.0;
      } else {
        weight_coefficient *= 4.0;
      }
    }

    std::vector<double> point(coordinates.size());
    for (size_t dim_index = 0; dim_index < coordinates.size(); ++dim_index) {
      point[dim_index] = input.left_bounds[dim_index] + coordinates[dim_index] * step_sizes[dim_index];
    }

    total_sum += weight_coefficient * IntegrandFunction(point);
    return;
  }

  for (int node_index = 0; node_index <= partitions; ++node_index) {
    coordinates[dimension] = node_index;
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

  std::vector<double> step_sizes(dimensions);
  for (int dim_index = 0; dim_index < dimensions; ++dim_index) {
    step_sizes[dim_index] = (input.right_bounds[dim_index] - input.left_bounds[dim_index]) / partitions;
  }

  std::vector<double> coordinates(dimensions, 0.0);
  double total_sum = 0.0;
  RecursiveIntegration(0, coordinates, step_sizes, total_sum);

  double volume_element = 1.0;
  for (int dim_index = 0; dim_index < dimensions; ++dim_index) {
    volume_element *= step_sizes[dim_index];
  }

  GetOutput() = total_sum * volume_element / std::pow(3.0, dimensions);
  return true;
}

bool AshihminDCalculateIntegralsBySimpsonSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace ashihmin_d_calculate_integrals_by_simpson
