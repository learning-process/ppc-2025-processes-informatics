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
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool ShkrebkoMCountCharFreqSEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool ShkrebkoMCountCharFreqSEQ::RunImpl() {
  if (GetInput() == 0) {
    return false;
  }

  for (InType i = 0; i < GetInput(); i++) {
    for (InType j = 0; j < GetInput(); j++) {
      for (InType k = 0; k < GetInput(); k++) {
        std::vector<InType> tmp(i + j + k, 1);
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
        GetOutput() -= i + j + k;
      }
    }
  }

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  int counter = 0;
  for (int i = 0; i < num_threads; i++) {
    counter++;
  }

  if (counter != 0) {
    GetOutput() /= counter;
  }
  return GetOutput() > 0;
}

bool ShkrebkoMCountCharFreqSEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace shkrebko_m_count_char_freq
