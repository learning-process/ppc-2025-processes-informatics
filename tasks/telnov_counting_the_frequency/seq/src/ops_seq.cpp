#include "telnov_counting_the_frequency/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "telnov_counting_the_frequency/common/include/common.hpp"
#include "util/include/util.hpp"

namespace telnov_counting_the_frequency {

TelnovCountingTheFrequencySEQ::TelnovCountingTheFrequencySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TelnovCountingTheFrequencySEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool TelnovCountingTheFrequencySEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool TelnovCountingTheFrequencySEQ::RunImpl() {
  const std::string &s = g_data_string;
  long long cnt = 0;
  for (char c : s) {
    if (c == 'X') {
      cnt++;
    }
  }

  GetOutput() = cnt;
  return true;
}

bool TelnovCountingTheFrequencySEQ::PostProcessingImpl() {
  return GetOutput() == GetInput();
}

}  // namespace telnov_counting_the_frequency
