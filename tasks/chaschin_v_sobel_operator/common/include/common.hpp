#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace chaschin_v_sobel_operator {

struct Pixel {
  uint8_t r, g, b;
};

using InType = std::tuple<std::vector<std::vector<Pixel>>, int, int>;
using OutType = std::vector<std::vector<Pixel>>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace chaschin_v_sobel_operator
