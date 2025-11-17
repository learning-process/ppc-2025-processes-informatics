#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace chaschin_v_max_for_each_row {

using InType = int;
using OutType = int;
int f = 1;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chaschin_v_max_for_each_row
