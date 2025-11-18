#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace ovsyannikov_n_num_mistm_in_two_str
