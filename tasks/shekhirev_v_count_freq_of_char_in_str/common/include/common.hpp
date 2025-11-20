#pragma once

#include <cstring>
#include <iostream>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace shekhirev_v_char_freq_seq {

struct InputData {
  std::string str;
  char target;

  InputData() : str(), target(0) {}

  InputData(std::string s, char t) : str(std::move(s)), target(t) {}

  bool operator==(const InputData &other) const {
    return str == other.str && target == other.target;
  }

  // Определяем operator<< как friend ПРЯМО ВНУТРИ структуры.
  // Это позволяет ADL (Argument Dependent Lookup) находить его мгновенно из любого места.
  friend std::ostream &operator<<(std::ostream &os, const InputData &data) {
    os << "{ str: \"" << data.str << "\", target: '" << data.target << "' }";
    return os;
  }
};

// (Старое определение operator<< удали отсюда, оно теперь внутри структуры)

using InType = InputData;
using OutType = int;
using BaseTask = ppc::task::Task<InType, OutType>;
using TestType = std::tuple<InputData, OutType, std::string>;

}  // namespace shekhirev_v_char_freq_seq
