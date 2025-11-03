#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace baldin_a_word_count {

using InType = std::string;
using OutType = long long;
using TestType = std::tuple<std::string, long long>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace baldin_a_word_count
