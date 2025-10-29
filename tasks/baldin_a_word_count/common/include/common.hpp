#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace baldin_a_word_count {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<std::string, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace baldin_a_word_count
