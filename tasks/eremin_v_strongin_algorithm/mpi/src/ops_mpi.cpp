#include "eremin_v_strongin_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"

namespace eremin_v_strongin_algorithm {

EreminVStronginAlgorithmMPI::EreminVStronginAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool EreminVStronginAlgorithmMPI::ValidationImpl() {
  auto &input = GetInput();
  return (std::get<0>(input) < std::get<1>(input)) && (std::get<2>(input) > 0) && (std::get<2>(input) <= 100000000) &&
         (std::get<0>(input) >= -1e9) && (std::get<0>(input) <= 1e9) && (std::get<1>(input) >= -1e9) &&
         (std::get<1>(input) <= 1e9) && (GetOutput() == 0);
}

bool EreminVStronginAlgorithmMPI::PreProcessingImpl() {
  return true;
}

bool EreminVStronginAlgorithmMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double lower_bound = 0.0;
  double upper_bound = 0.0;
  int steps = 0;
  auto &input = GetInput();
  auto function = std::get<3>(input);

  if (rank == 0) {
    auto &input = GetInput();
    lower_bound = std::get<0>(input);
    upper_bound = std::get<1>(input);
    steps = std::get<2>(input);
  }

  MPI_Bcast(&lower_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&upper_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> points = {lower_bound, upper_bound};
  std::vector<double> function_values = {function(lower_bound), function(upper_bound)};

  const double r_constant = 2.0;

  for (int step = 0; step < steps; step++) {
    int num_points = points.size();

    double local_max_slope = 0.0;
    for (int i = rank; i < num_points; i += size) {
      if (i == 0) {
        continue;
      }
      double slope = std::abs(function_values[i] - function_values[i - 1]) / (points[i] - points[i - 1]);
      if (slope > local_max_slope) {
        local_max_slope = slope;
      }
    }

    double global_max_slope = 0.0;
    MPI_Allreduce(&local_max_slope, &global_max_slope, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double max_slope_between_points = (global_max_slope > 0.0) ? r_constant * global_max_slope : 1.0;

    double local_max_characteristic = -1e18;
    int local_best_interval_idx = -1;

    for (int i = rank; i < num_points; i += size) {
      if (i == 0) {
        continue;
      }
      double interval_length = points[i] - points[i - 1];
      double function_diff = function_values[i] - function_values[i - 1];
      double characteristic = max_slope_between_points * interval_length +
                              (function_diff * function_diff) / (max_slope_between_points * interval_length) -
                              2.0 * (function_values[i] + function_values[i - 1]);
      if (characteristic > local_max_characteristic) {
        local_max_characteristic = characteristic;
        local_best_interval_idx = i;
      }
    }

    struct {
      double val;
      int idx;
    } local_data{local_max_characteristic, local_best_interval_idx}, global_data;
    MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    int best_interval_idx = global_data.idx;

    if (best_interval_idx > 0) {
      double new_point = 0.5 * (points[best_interval_idx] + points[best_interval_idx - 1]) -
                         (function_values[best_interval_idx] - function_values[best_interval_idx - 1]) /
                             (2.0 * max_slope_between_points);
      double new_point_value = function(new_point);

      points.insert(points.begin() + best_interval_idx, new_point);
      function_values.insert(function_values.begin() + best_interval_idx, new_point_value);
    }
  }
  double local_min_value = *std::min_element(function_values.begin(), function_values.end());
  double global_min_value = 0.0;
  MPI_Allreduce(&local_min_value, &global_min_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min_value;

  return true;
}

bool EreminVStronginAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_strongin_algorithm
