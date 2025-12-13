#pragma once

#include <tuple>

#include "task/include/task.hpp"

namespace telnov_strongin_algorithm {

struct StronginInput {
  double a;
  double b;
  double eps;
};

using InType = StronginInput;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace telnov_strongin_algorithm
