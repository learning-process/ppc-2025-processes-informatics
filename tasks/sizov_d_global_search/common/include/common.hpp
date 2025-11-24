#pragma once

#include <functional>
#include <string>

#include "task/include/task.hpp"

namespace sizov_d_global_search {

using Function = std::function<double(double)>;

struct Problem {
  Function func;
  double left = 0.0;
  double right = 0.0;
  double accuracy = 1e-3;    // желаемая точность локализации минимума
  double reliability = 2.0;  // r - коэффициент надежности
  int max_iterations = 1000;
};

struct Solution {
  double argmin = 0.0;
  double value = 0.0;
  int iterations = 0;
  bool converged = false;
};

struct TestCase {
  std::string name;
  Problem problem;
  double brute_force_step = 1e-3;
  double tolerance = 1e-3;
};

using InType = Problem;
using OutType = Solution;
using TestType = TestCase;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sizov_d_global_search
