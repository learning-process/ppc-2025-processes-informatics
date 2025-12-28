#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#include "task/include/task.hpp"

namespace kruglova_a_2d_multistep_par_opt {

struct InType {
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  double eps;
  int max_iters;
};

struct OutType {
  double x;
  double y;
  double f_value;
};

struct Interval1D {
  double a, b;
  double f_a, f_b;
  double characteristic;
  int iteration;
};

struct Trial {
  double point;
  double value;
};

using TestType = std::tuple<std::string, InType>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline double ObjectiveFunction(double x, double y) {
  constexpr double A = 10.0;
  constexpr double n = 2.0;
  return A * n + x * x + y * y - A * (std::cos(2.0 * M_PI * x) + std::cos(2.0 * M_PI * y));
}

}  // namespace kruglova_a_2d_multistep_par_opt
