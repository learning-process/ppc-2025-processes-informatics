#pragma once

#include <cstddef>
#include <functional>

#include "task/include/task.hpp"

namespace global_search_strongin {

struct StronginInput {
  double left = 0.0;
  double right = 1.0;

  double epsilon = 1e-3;

  double reliability = 2.0;

  int max_iterations = 1000;

  std::function<double(double)> objective;
};

struct StronginResult {
  double best_point = 0.0;
  double best_value = 0.0;
  int iterations = 0;
};

struct SamplePoint {
  double x = 0.0;
  double value = 0.0;
};

using InType = StronginInput;
using OutType = StronginResult;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace global_search_strongin
