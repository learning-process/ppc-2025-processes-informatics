#pragma once

#include <iostream>  // Добавили
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace shekhirev_v_char_freq_seq {

struct InputData {
  std::string str;
  char target;

  bool operator==(const InputData &other) const {
    return str == other.str && target == other.target;
  }
};

inline std::ostream &operator<<(std::ostream &os, const InputData &data) {
  os << "{ str: \"" << data.str << "\", target: '" << data.target << "' }";
  return os;
}

using InType = InputData;
using OutType = int;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<InputData, OutType, std::string>;

}  // namespace shekhirev_v_char_freq_seq
