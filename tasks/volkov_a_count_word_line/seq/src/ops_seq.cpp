#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

#include <cstddef>

#include "volkov_a_count_word_line/common/include/common.hpp"

namespace volkov_a_count_word_line {

static size_t count_words(const char *data, size_t n) {
  auto is_token_char = [](char c) -> bool {
    const bool is_alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    const bool is_digit = (c >= '0' && c <= '9');
    const bool is_special = (c == '-' || c == '_');
    return is_alpha || is_digit || is_special;
  };

  size_t count = 0;
  size_t i = 0;
  while (i < n) {
    while (i < n && !is_token_char(data[i])) {
      i++;
    }
    if (i < n) {
      count++;
      while (i < n && is_token_char(data[i])) {
        i++;
      }
    }
  }
  return count;
}

VolkovACountWordLineSEQ::VolkovACountWordLineSEQ(const InType &in) {
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
  const char *data = reinterpret_cast<const char *>(GetInput().data());
  const size_t size = GetInput().size();
  GetOutput() = count_words(data, size);
  return true;
}

bool VolkovACountWordLineSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_count_word_line
