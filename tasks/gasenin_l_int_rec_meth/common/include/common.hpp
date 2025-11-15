#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace gasenin_l_int_rec_meth {

using InType = std::pair<std::string, std::string>;
using OutType = int;  // -1: первая меньше, 0: равны, 1: первая больше
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gasenin_l_int_rec_meth
