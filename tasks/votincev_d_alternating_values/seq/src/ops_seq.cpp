#include "votincev_d_alternating_values/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "util/include/util.hpp"
#include "votincev_d_alternating_values/common/include/common.hpp"

namespace votincev_d_alternating_values {

VotincevDAlternatingValuesSEQ::VotincevDAlternatingValuesSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VotincevDAlternatingValuesSEQ::ValidationImpl() {
  return !(GetInput().empty());
}

bool VotincevDAlternatingValuesSEQ::PreProcessingImpl() {
  v = GetInput();
  return true;
}

bool VotincevDAlternatingValuesSEQ::RunImpl() {
  // v = GetInput();  // напишу в начало на всякий (проверка)
  int allSwaps = 0;
  for (size_t j = 1; j < v.size(); j++) {
    if ((v[j - 1] < 0 && v[j] >= 0) || (v[j - 1] >= 0 && v[j] < 0)) {
      allSwaps++;
    }
  }
  GetOutput() = allSwaps;
  return true;
}

bool VotincevDAlternatingValuesSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_alternating_values
