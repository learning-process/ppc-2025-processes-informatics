#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

#include <string>

namespace volkov_a_count_word_line {

VolkovACountWordLineSEQ::VolkovACountWordLineSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VolkovACountWordLineSEQ::ValidationImpl() {
  return true;
}

bool VolkovACountWordLineSEQ::PreProcessingImpl() {
  return true;
}

bool VolkovACountWordLineSEQ::RunImpl() {
  const std::string& data = GetInput();

  auto is_token_char = [](char c) -> bool {
    const bool is_alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    const bool is_digit = (c >= '0' && c <= '9');
    const bool is_special = (c == '-' || c == '_');
    return is_alpha || is_digit || is_special;
  };

  int counter = 0;
  size_t i = 0;
  const size_t n = data.size();

  while (i < n) {
    while (i < n && !is_token_char(data[i])) {
      i++;
    }

    if (i < n) {
      counter++;
      while (i < n && is_token_char(data[i])) {
        i++;
      }
    }
  }

  GetOutput() = counter;
  return true;
}

bool VolkovACountWordLineSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_count_word_line