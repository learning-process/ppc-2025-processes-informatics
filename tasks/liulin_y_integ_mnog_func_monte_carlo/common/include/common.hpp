#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

struct TaskInput {
  double x_min, x_max;                      // Границы по x
  double y_min, y_max;                      // Границы по y
  std::function<double(double, double)> f;  // f(x, y)
  long long num_points;                     // Количество случайных точек
};

using InType = TaskInput;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace liulin_y_integ_mnog_func_monte_carlo
