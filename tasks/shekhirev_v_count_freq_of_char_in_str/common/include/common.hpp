#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shekhirev_v_char_freq_seq {

struct InputData {
  std::string str;
  char target;
};

using InType = InputData;
using OutType = int;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<InputData, OutType, std::string>;

}  // namespace shekhirev_v_char_freq_seq
