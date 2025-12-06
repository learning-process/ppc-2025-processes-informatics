#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace levonychev_i_mult_matrix_vec {

using InType = std::tuple<std::vector<int64_t>, int, int, std::vector<int64_t>>;
using OutType = std::vector<int64_t>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace levonychev_i_mult_matrix_vec
