#include "krykov_e_word_count/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "krykov_e_word_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace krykov_e_word_count {

KrykovEWordCountSEQ::KrykovEWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrykovEWordCountSEQ::ValidationImpl() {
  return (GetOutput() == 0);
}

bool KrykovEWordCountSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KrykovEWordCountSEQ::RunImpl() {
  const std::string &text = GetInput();
  if (text.empty()) {
    return false;
  }

  bool in_word = false;
  int word_count = 0;

  for (char c : text) {
    if (std::isspace(c) || std::ispunct(c)) {
      if (in_word) {
        word_count++;
        in_word = false;
      }
    } else {
      in_word = true;
    }
  }

  if (in_word) {
    word_count++;
  }

  GetOutput() = word_count;
  return GetOutput() >= 0;
}

bool KrykovEWordCountSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace krykov_e_word_count
