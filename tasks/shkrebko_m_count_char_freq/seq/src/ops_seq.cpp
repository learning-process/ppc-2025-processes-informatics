#include "shkrebko_m_count_char_freq/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "shkrebko_m_count_char_freq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkrebko_m_count_char_freq {

ShkrebkoMCountCharFreqSEQ::ShkrebkoMCountCharFreqSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkrebkoMCountCharFreqSEQ::ValidationImpl() {
  const auto &input_data = GetInput();
  return !input_data.first.empty();
}

bool ShkrebkoMCountCharFreqSEQ::PreProcessingImpl() {
  const auto &input_data = GetInput();
  input_text_ = input_data.first;
  target_char_ = input_data.second;
  result_count_ = 0;
  return true;
}

bool ShkrebkoMCountCharFreqSEQ::RunImpl() {
  result_count_ = std::count(input_text_.begin(), input_text_.end(), target_char_);
  return true;
}

bool ShkrebkoMCountCharFreqSEQ::PostProcessingImpl() {
  GetOutput() = result_count_;
  return true;
}

}  // namespace shkrebko_m_count_char_freq
