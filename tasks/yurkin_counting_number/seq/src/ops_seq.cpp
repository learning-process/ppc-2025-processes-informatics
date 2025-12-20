#include "yurkin_counting_number/seq/include/ops_seq.hpp"

#include <cctype>

#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

YurkinCountingNumberSEQ::YurkinCountingNumberSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberSEQ::ValidationImpl() {
  return !GetInput().empty() && GetOutput() == 0;
}

bool YurkinCountingNumberSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberSEQ::RunImpl() {
  (void)GetOutput();
  const InType &data = GetInput();
  int count = 0;
  for (char c : data) {
    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
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
