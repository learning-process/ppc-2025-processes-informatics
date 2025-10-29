#include "akimov_i_words_string_count/seq/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>

#include "akimov_i_words_string_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace akimov_i_words_string_count {

AkimovIWordsStringCountSEQ::AkimovIWordsStringCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkimovIWordsStringCountSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool AkimovIWordsStringCountSEQ::PreProcessingImpl() {
  input_buffer_ = GetInput();
  word_count_ = 0;
  space_count_ = 0;
  return true;
}

bool AkimovIWordsStringCountSEQ::RunImpl() {
  if (input_buffer_.empty()) {
    return false;
  }

  for (char c : input_buffer_) {
    if (c == ' ') {
      ++space_count_;
    }
  }

  bool in_word = false;
  word_count_ = 0;
  for (char c : input_buffer_) {
    if (c != ' ' && !in_word) {
      in_word = true;
      ++word_count_;
    } else if (c == ' ' && in_word) {
      in_word = false;
    }
  }

  return true;
}

bool AkimovIWordsStringCountSEQ::PostProcessingImpl() {
  GetOutput() = word_count_;
  return true;
}

}  // namespace akimov_i_words_string_count
