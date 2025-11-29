#include "volkov_a_count_word_line/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>

#include "volkov_a_count_word_line/common/include/common.hpp"

namespace volkov_a_count_word_line {
namespace {

bool IsTokenChar(char c) {
  const bool is_alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  const bool is_digit = (c >= '0' && c <= '9');
  const bool is_special = (c == '-' || c == '_');
  return is_alpha || is_digit || is_special;
}

int CountWords(const char *data, size_t n) {
  enum class State : std::uint8_t { kInWord, kInSeparator };
  int word_count = 0;

  State current_state = State::kInSeparator;

  for (size_t i = 0; i < n; ++i) {
    const char c = data[i];
    if (current_state == State::kInSeparator) {
      if (IsTokenChar(c)) {
        word_count++;
        current_state = State::kInWord;
      }
    } else {
      if (!IsTokenChar(c)) {
        current_state = State::kInSeparator;
      }
    }
  }
  return word_count;
}

}  // namespace

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
  GetOutput() = CountWords(data, size);
  return true;
}

bool VolkovACountWordLineSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_count_word_line
