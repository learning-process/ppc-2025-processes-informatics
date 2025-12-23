#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace ovsyannikov_n_shell_batcher {
using InType = std::vector<int>;
using OutType = std::vector<int>;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<InType, OutType, std::string>;
}  // namespace ovsyannikov_n_shell_batcher