#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dorofeev_i_scatter {

using InType = std::vector<double>;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dorofeev_i_scatter
