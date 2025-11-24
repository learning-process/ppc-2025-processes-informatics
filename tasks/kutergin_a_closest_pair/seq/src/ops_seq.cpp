#include "kutergin_a_closest_pair/seq/include/ops_seq.hpp"

#include <cstdlib>
#include <ranges>
#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"

namespace kutergin_a_closest_pair {

KuterginAClosestPairSEQ::KuterginAClosestPairSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = -1;
}

bool KuterginAClosestPairSEQ::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool KuterginAClosestPairSEQ::PreProcessingImpl() {
  return true;
}

bool KuterginAClosestPairSEQ::RunImpl() {
  if (data_.size() < 2) {
    GetOutput() = -1;
    return true;
  }

  int min_diff = std::numeric_limits<int>::max();
  int min_idx = -1;

  for (size_t i = 0; i < data_.size() - 1; ++i) {
    int diff = std::abs(data_[i + 1] - data_[i]);
    if (diff < min_diff) {
      min_diff = diff;
      min_idx = static_cast<int>(i);
    }
  }

  GetOutput() = min_idx;
  return true;
}

bool KuterginAClosestPairSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_a_closest_pair
