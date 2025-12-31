#include "eremin_v_strongin_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"

namespace eremin_v_strongin_algorithm {

EreminVStronginAlgorithmSEQ::EreminVStronginAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool EreminVStronginAlgorithmSEQ::ValidationImpl() {
  auto &input = GetInput();
  return (std::get<0>(input) < std::get<1>(input)) && (std::get<2>(input) > 0) && (std::get<2>(input) <= 100000000) &&
         (std::get<0>(input) >= -1e9) && (std::get<0>(input) <= 1e9) && (std::get<1>(input) >= -1e9) &&
         (std::get<1>(input) <= 1e9) && (GetOutput() == 0);
}

bool EreminVStronginAlgorithmSEQ::PreProcessingImpl() {
  return true;
}

bool EreminVStronginAlgorithmSEQ::RunImpl() {
  auto &input = GetInput();
  double lower_bound = std::get<0>(input);
  double upper_bound = std::get<1>(input);
  int steps = std::get<2>(input);
  auto function = std::get<3>(input);

  std::vector<double> points = {lower_bound, upper_bound};
  std::vector<double> function_values = {function(lower_bound), function(upper_bound)};

  const double r_constant = 2.0;

  for (int step = 0; step < steps; step++) {
    int num_points = points.size();

    double max_slope = 0.0;
    for (int i = 1; i < num_points; i++) {
      double slope = std::abs(function_values[i] - function_values[i - 1]) / (points[i] - points[i - 1]);
      if (slope > max_slope) {
        max_slope = slope;
      }
    }
    double max_slope_between_points = (max_slope > 0.0) ? r_constant * max_slope : 1.0;

    double max_characteristic = -1e18;
    int best_interval_idx = -1;
    for (int i = 1; i < num_points; i++) {
      double interval_length = points[i] - points[i - 1];
      double function_diff = function_values[i] - function_values[i - 1];
      double characteristic = max_slope_between_points * interval_length +
                              (function_diff * function_diff) / (max_slope_between_points * interval_length) -
                              2.0 * (function_values[i] + function_values[i - 1]);
      if (characteristic > max_characteristic) {
        max_characteristic = characteristic;
        best_interval_idx = i;
      }
    }

    if (best_interval_idx > 0) {
      double new_point = 0.5 * (points[best_interval_idx] + points[best_interval_idx - 1]) -
                         (function_values[best_interval_idx] - function_values[best_interval_idx - 1]) /
                             (2.0 * max_slope_between_points);
      double new_point_value = function(new_point);

      points.insert(points.begin() + best_interval_idx, new_point);
      function_values.insert(function_values.begin() + best_interval_idx, new_point_value);
    }
  }

  auto min_it = std::min_element(function_values.begin(), function_values.end());
  int min_index = std::distance(function_values.begin(), min_it);
  double min_point = points[min_index];
  GetOutput() = min_point;

  return true;
}

bool EreminVStronginAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_strongin_algorithm
