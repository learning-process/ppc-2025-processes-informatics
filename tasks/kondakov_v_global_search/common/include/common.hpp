#pragma once

#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kondakov_v_global_search {

struct Params {
  std::function<double(double)> func = nullptr;
  double left = 0.0;
  double right = 0.0;
  double accuracy = 1e-6;
  double reliability = 1.0;
  int max_iterations = 1000;
};

struct Solution {
  double argmin = 0.0;
  double value = 0.0;
  int iterations = 0;
  bool converged = false;
};

using InType = Params;
using OutType = Solution;
using TestType = std::tuple<Params, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kondakov_v_global_search
