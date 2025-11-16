#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace potashnik_m_char_freq {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace potashnik_m_char_freq
