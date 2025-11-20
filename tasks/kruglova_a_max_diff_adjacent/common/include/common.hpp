#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"
#include "util/include/perf_test_util.hpp"

namespace kruglova_a_max_diff_adjacent {

using InType = std::vector<float>;
using OutType = float;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kruglova_a_max_diff_adjacent
