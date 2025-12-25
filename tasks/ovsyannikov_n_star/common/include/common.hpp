#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace ovsyannikov_n_star {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ovsyannikov_n_star
