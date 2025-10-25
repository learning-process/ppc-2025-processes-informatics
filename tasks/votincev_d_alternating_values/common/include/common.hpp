#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace votincev_d_alternating_values {

using InType = std::vector<double>;  // std::vector<double>
using OutType = int;
using TestType = std::vector<double>;  // std::vector<double>
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace votincev_d_alternating_values
