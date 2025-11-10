#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace shvetsova_k_max_diff_neig_vec {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shvetsova_k_max_diff_neig_vec
