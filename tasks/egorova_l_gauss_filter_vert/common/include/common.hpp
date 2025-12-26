#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include "task/include/task.hpp"

namespace egorova_l_gauss_filter_vert {

struct Image {
    std::vector<uint8_t> data;
    int rows = 0;
    int cols = 0;
    int channels = 1;
};

using InType = Image;
using OutType = Image;
using TestType = std::tuple<int, int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace egorova_l_gauss_filter_vert