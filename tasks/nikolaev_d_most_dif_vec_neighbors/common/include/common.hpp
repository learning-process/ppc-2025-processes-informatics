#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nikolaev_d_most_dif_vec_neighbors
