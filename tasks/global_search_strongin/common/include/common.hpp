#pragma once

#include <functional>
#include <vector>

#include "task/include/task.hpp"

namespace global_search_strongin {

struct StronginInput {
  double left = 0.0;
  double right = 0.0;
  double reliability = 2.0;
  double epsilon = 1e-3;
  int max_iterations = 1000;
  std::function<double(double)> objective;
};

struct StronginResult {
  double best_point = 0.0;
  double best_value = 0.0;
  int iterations = 0;
};

using InType = StronginInput;
using OutType = StronginResult;
using BaseTask = ppc::task::Task<InType, OutType>;

struct SamplePoint {
  double x = 0.0;
  double value = 0.0;
};

}  // namespace global_search_strongin
