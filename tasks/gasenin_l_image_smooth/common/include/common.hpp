#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace gasenin_l_image_smooth {

struct TaskData {
  std::vector<uint8_t> data;
  int width;
  int height;
  int kernel_size;

  bool operator==(const TaskData &other) const {
    return data == other.data && width == other.width && height == other.height;
  }
};

using InType = TaskData;
using OutType = TaskData;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline int clamp(int val, int min, int max) {
  if (val < min) {
    return min;
  }
  if (val > max) {
    return max;
  }
  return val;
}

}  // namespace gasenin_l_image_smooth
