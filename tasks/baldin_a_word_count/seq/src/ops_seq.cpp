#include "baldin_a_word_count/seq/include/ops_seq.hpp"

#include "baldin_a_word_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_word_count {

BaldinAWordCountSEQ::BaldinAWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaldinAWordCountSEQ::ValidationImpl() {
  return true;
}

bool BaldinAWordCountSEQ::PreProcessingImpl() {
  return true;
}

bool BaldinAWordCountSEQ::RunImpl() {
  int count = 0;
  bool in_word = false;
  for (char c : GetInput()) {
    if (std::isalnum(c) || c == '-' || c == '_') {
      if (!in_word) {
        in_word = true;
        count++;
      }
    } else {
      in_word = false;
    }
  }

  GetOutput() = count;
  return true;
}

bool BaldinAWordCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_word_count
