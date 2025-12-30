#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dolov_v_qsort_batcher {

using InType = std::vector<double>;
using OutType = std::vector<double>;

using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dolov_v_qsort_batcher
