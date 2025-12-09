#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace telnov_transfer_one_all {

using TestType = std::tuple<int, std::string>;

template<typename T>
using BaseTask = ppc::task::Task<std::vector<T>, std::vector<T>>;

}  // namespace telnov_transfer_one_all
