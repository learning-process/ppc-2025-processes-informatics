#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace shkrebko_m_count_char_freq {

using InType = std::pair<std::string, char>;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shkrebko_m_count_char_freq
