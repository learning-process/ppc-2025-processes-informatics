#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

using InType = std::vector<int>;
using OutType = int;
using TestType = int;  // std::tuple<std::tuple<std::vector<int>, int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace konstantinov_s_elem_vec_sign_change_count
