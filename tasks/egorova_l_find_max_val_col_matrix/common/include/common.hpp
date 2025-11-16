#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace egorova_l_find_max_val_col_matrix {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egorova_l_find_max_val_col_matrix
