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
  GetOutput() = 0.0;
}

bool EreminVStronginAlgorithmSEQ::ValidationImpl() {
  auto &input = GetInput();

  double lower_bound = std::get<0>(input);
  double upper_bound = std::get<1>(input);
  double epsilon = std::get<2>(input);
  int max_iters = std::get<3>(input);

  return (lower_bound < upper_bound) && (epsilon > 0.0 && epsilon <= (upper_bound - lower_bound)) &&
         (max_iters > 0 && max_iters <= 100000000) && (lower_bound >= -1e9 && lower_bound <= 1e9) &&
         (upper_bound >= -1e9 && upper_bound <= 1e9) && (GetOutput() == 0);
}

bool EreminVStronginAlgorithmSEQ::PreProcessingImpl() {
  auto &input = GetInput();
    lower_bound = std::get<0>(input);
    upper_bound = std::get<1>(input);
    epsilon = std::get<2>(input);
    max_iterations = std::get<3>(input);

  objective_function = std::get<4>(input);

  search_points.push_back(lower_bound);
  search_points.push_back(upper_bound);
  function_values.push_back(objective_function(lower_bound));
  function_values.push_back(objective_function(upper_bound));
  return true;
}

bool EreminVStronginAlgorithmSEQ::RunImpl() {
  int current_iteration = 0;
  double r_coefficient = 2.0;
  double max_interval_width = upper_bound - lower_bound;

  while (max_interval_width > epsilon && current_iteration < max_iterations) {
    ++current_iteration;

    double lipschitz_estimate = 0.0;
    for (std::size_t i = 1; i < search_points.size(); ++i) {
      double interval_width = search_points[i] - search_points[i - 1];
      double value_difference = std::abs(function_values[i] - function_values[i - 1]);
      double current_slope = value_difference / interval_width;
      if (current_slope > lipschitz_estimate) {
        lipschitz_estimate = current_slope;
      }
    }

    double m_parameter = (lipschitz_estimate > 0.0) ? r_coefficient * lipschitz_estimate : 1.0;

    double max_characteristic = -1e18;
    std::size_t best_interval_index = 1;

    for (std::size_t i = 1; i < search_points.size(); ++i) {
      double interval_width = search_points[i] - search_points[i - 1];
      double value_difference = function_values[i] - function_values[i - 1];

      double characteristic = (m_parameter * interval_width) +
                              (value_difference * value_difference) / (m_parameter * interval_width) -
                              (2.0 * (function_values[i] + function_values[i - 1]));

      if (characteristic > max_characteristic) {
        max_characteristic = characteristic;
        best_interval_index = i;
      }
    }

    double left_point = search_points[best_interval_index - 1];
    double right_point = search_points[best_interval_index];
    double left_value = function_values[best_interval_index - 1];
    double right_value = function_values[best_interval_index];

    double new_point = 0.5 * (left_point + right_point) - (right_value - left_value) / (2.0 * m_parameter);

    if (new_point <= left_point || new_point >= right_point) {
      new_point = 0.5 * (left_point + right_point);
    }

    double new_value = objective_function(new_point);

    search_points.insert(search_points.begin() + static_cast<std::ptrdiff_t>(best_interval_index), new_point);
    function_values.insert(function_values.begin() + static_cast<std::ptrdiff_t>(best_interval_index), new_value);

    max_interval_width = 0.0;
    for (std::size_t i = 1; i < search_points.size(); ++i) {
      double current_width = search_points[i] - search_points[i - 1];
      if (current_width > max_interval_width) {
        max_interval_width = current_width;
      }
    }
  }

  GetOutput() = *std::ranges::min_element(function_values);
  return true;
}

bool EreminVStronginAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_strongin_algorithm
