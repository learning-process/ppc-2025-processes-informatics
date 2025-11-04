#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kutuzov_i_elem_vec_average {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kutuzov_i_elem_vec_average
