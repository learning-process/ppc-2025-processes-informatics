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
  // auto start_time = std::chrono::high_resolution_clock::now();

  int allSwaps = 0;
  for (size_t j = 1; j < v.size(); j++) {
    if ((v[j - 1] < 0 && v[j] >= 0) || (v[j - 1] >= 0 && v[j] < 0)) {
      allSwaps++;
    }
  }

  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  // std::cout << "SEQ_was_working:" << duration.count() / 1000000.0 << " seconds" << std::endl;
  GetOutput() = allSwaps;
  return true;
}

bool VotincevDAlternatingValuesSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_alternating_values
