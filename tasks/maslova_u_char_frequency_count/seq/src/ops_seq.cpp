#include "maslova_u_char_frequency_count/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include "maslova_u_char_frequency_count/common/include/common.hpp"

namespace maslova_u_char_frequency_count {

MaslovaUCharFrequencyCountSEQ::MaslovaUCharFrequencyCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MaslovaUCharFrequencyCountSEQ::ValidationImpl() {
  return true;
}

bool MaslovaUCharFrequencyCountSEQ::PreProcessingImpl() {
  return true;
}

bool MaslovaUCharFrequencyCountSEQ::RunImpl() {
  std::string &input_string = GetInput().first;
  char input_char = GetInput().second;  // получили данные
  size_t frequency_count = 0;

  for (const char c : input_string) {
    if (c == input_char) {
      frequency_count++;
    }
  }

  GetOutput() = frequency_count;  // отправили данные
  return true;
}

bool MaslovaUCharFrequencyCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_char_frequency_count
