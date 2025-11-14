#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace levonychev_i_min_val_rows_matrix {

using InType = std::tuple<std::vector<int>, size_t, size_t>;
using OutType = std::vector<int>;
using TestType = std::tuple<size_t, size_t>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace levonychev_i_min_val_rows_matrix
