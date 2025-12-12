#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace smyshlaev_a_gauss_filt {
struct ImageType {
  int width;
  int height;
  int channels;
  std::vector<uint8_t> data;
};

using InType = ImageType;
using OutType = ImageType;
using TestType = std::tuple<ImageType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace smyshlaev_a_gauss_filt
