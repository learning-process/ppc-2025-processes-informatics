#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

using InType = std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>;
using OutType = std::vector<int64_t>;
using TestType = std::tuple<InType, std::string, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
