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
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool KrykovEWordCountSEQ::PreProcessingImpl() {
  auto &input = GetInput();
  input.erase(input.begin(),
              std::find_if(input.begin(), input.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  input.erase(std::find_if(input.rbegin(), input.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              input.end());
  return true;
}

bool KrykovEWordCountSEQ::RunImpl() {
  const std::string &text = GetInput();
  if (text.empty()) {
    return false;
  }

  bool in_word = false;
  size_t word_count = 0;

  for (char c : text) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (in_word) {
        in_word = false;
      }
    } else {
      if (!in_word) {
        in_word = true;
        word_count++;
      }
    }
  }

  GetOutput() = word_count;
  return true;
}

bool KrykovEWordCountSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace krykov_e_word_count
