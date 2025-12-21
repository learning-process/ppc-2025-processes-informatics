#pragma once

#include <cmath>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace luzan_e_simps_int {

// n, a-b, c-d
using InType = std::tuple<int, std::tuple<int, int>, std::tuple<int, int>>;
using OutType = double;
// n, a-b, c-d
using TestType = std::tuple<int, std::tuple<int, int>, std::tuple<int, int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

const double EPSILON = 0.0001;

inline double f(double x, double y) {
  return pow(x, 5) / 5.0 + y * sin(y) + 2.0;
}

}  // namespace luzan_e_simps_int
