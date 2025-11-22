#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

using InType = std::tuple<long long, std::vector<long long>, std::vector<long long>>;
using OutType = std::vector<long long>;
using TestType = std::tuple<InType, std::string, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
