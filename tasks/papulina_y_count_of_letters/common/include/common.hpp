#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace papulina_y_count_of_letters {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<std::string, int>;  // патерн + длина строки
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace papulina_y_count_of_letters
