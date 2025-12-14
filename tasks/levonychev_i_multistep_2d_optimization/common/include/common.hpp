#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace levonychev_i_multistep_2d_optimization {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace levonychev_i_multistep_2d_optimization
