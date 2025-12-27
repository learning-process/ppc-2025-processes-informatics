#include "agafonov_i_sentence_count/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cctype>

namespace agafonov_i_sentence_count {

SentenceCountSEQ::SentenceCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SentenceCountSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool SentenceCountSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SentenceCountSEQ::RunImpl() {
  const std::string& text = GetInput();
  int count = 0;
  bool in_sentence = false;

  for (size_t i = 0; i < text.length(); ++i) {
    char c = text[i];
    
    if (std::isalpha(c) || std::isdigit(c)) {
      in_sentence = true;
    } else if (c == '.' && in_sentence) {
      count++;
      in_sentence = false;
    } else if ((c == '!' || c == '?') && in_sentence) {
      count++;
      in_sentence = false;
    }
  }

  if (in_sentence) {
    count++;
  }

  GetOutput() = count;
  return true;
}

bool SentenceCountSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace agafonov_i_sentence_count