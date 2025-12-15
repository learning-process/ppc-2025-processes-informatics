#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace egashin_k_iterative_simple {

struct InputData {
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> x0;
  double tolerance;
  int max_iterations;
};

using InType = InputData;
using OutType = std::vector<double>;

using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egashin_k_iterative_simple
