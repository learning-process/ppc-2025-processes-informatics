#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kruglova_a_max_diff_adjacent {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kruglova_a_max_diff_adjacent
