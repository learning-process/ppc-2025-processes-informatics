#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace maslova_u_char_frequency_count {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace maslova_u_char_frequency_count
