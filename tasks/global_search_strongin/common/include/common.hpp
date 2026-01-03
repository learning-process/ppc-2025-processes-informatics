#pragma once

#include <functional>
#include <tuple>

#include "task/include/task.hpp"

namespace global_search_strongin {

struct SamplePoint {
  double x = 0.0;
  double value = 0.0;
};

using InType = std::tuple<double, double, double, int, std::function<double(double)>>;
using OutType = double;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace global_search_strongin
