#pragma once

#include <string>
#include <utility>

#include "task/include/task.hpp"

namespace ovsyannikov_n_num_mistm_in_two_str { 
using InType = std::pair<std::string, std::string>; // Входные данные: пара строк
using OutType = int; // Выходные данные: количество несовпадений 
using BaseTask = ppc::task::Task<InType, OutType>;
}  // namespace ovsyannikov_n_num_mistm_in_two_str