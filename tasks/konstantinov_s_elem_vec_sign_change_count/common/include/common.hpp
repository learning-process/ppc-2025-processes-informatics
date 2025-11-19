#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace konstantinov_s_elem_vec_sign_change_count {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace konstantinov_s_elem_vec_sign_change_count
