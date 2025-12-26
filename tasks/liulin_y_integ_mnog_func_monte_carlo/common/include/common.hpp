#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "task/include/task.hpp"

namespace liulin_y_integ_mnog_func_monte_carlo {

struct TaskInput {
  double x_min = 0.0, x_max = 0.0;          // Границы по x
  double y_min = 0.0, y_max = 0.0;          // Границы по y
  std::function<double(double, double)> f;  // f(x, y)
  int64_t num_points = 0;                   // Количество случайных точек
};

using InType = TaskInput;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace liulin_y_integ_mnog_func_monte_carlo
