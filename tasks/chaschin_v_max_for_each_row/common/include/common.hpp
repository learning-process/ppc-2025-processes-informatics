#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace chaschin_v_max_for_each_row {

using InType = std::Vector<std::Vector<float>>;
using OutType = std::Vector<float>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chaschin_v_max_for_each_row
