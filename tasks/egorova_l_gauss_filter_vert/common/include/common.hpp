#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace egorova_l_gauss_filter_vert {

struct Image {
  int rows = 0;
  int cols = 0;
  int channels = 0;
  std::vector<uint8_t> data;
};

using InType = Image;
using OutType = Image;
using TestType = std::tuple<int, int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egorova_l_gauss_filter_vert