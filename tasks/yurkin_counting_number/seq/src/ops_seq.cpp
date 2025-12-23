#include "yurkin_counting_number/seq/include/ops_seq.hpp"

#include <cctype>

namespace yurkin_counting_number {

YurkinCountingNumberSEQ::YurkinCountingNumberSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberSEQ::ValidationImpl() {
  return GetOutput() == 0;
}

bool YurkinCountingNumberSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberSEQ::RunImpl() {
  const auto &input = GetInput();
  int count = 0;

  for (unsigned char c : input) {
    if (std::isalpha(c) != 0) {
      ++count;
    }
  }

  GetOutput() = count;
  return true;
}

bool YurkinCountingNumberSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_counting_number
