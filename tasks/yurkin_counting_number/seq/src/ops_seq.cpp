#include "yurkin_counting_number/seq/include/ops_seq.hpp"

#include <cctype>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

YurkinCountingNumberSEQ::YurkinCountingNumberSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberSEQ::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinCountingNumberSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberSEQ::RunImpl() {
  int input = GetInput();
  int count = input;
  GetOutput() = count;
  return true;
}

bool YurkinCountingNumberSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_counting_number
