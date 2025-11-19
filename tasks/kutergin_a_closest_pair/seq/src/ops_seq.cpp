#include "kutergin_a_closest_pair/seq/include/ops_seq.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kutergin_a_closest_pair {

KuterginAClosestPairSEQ::KuterginAClosestPairSEQ(const InType &in) : data_() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = -1;
  data_ = in;
}

bool KuterginAClosestPairSEQ::ValidationImpl() {
  return true;
}

bool KuterginAClosestPairSEQ::PreProcessingImpl() {
  return true;
}

bool KuterginAClosestPairSEQ::RunImpl() {
  const auto &v = GetInput();
  int n = static_cast<int>(v.size());

  if (n < 2) {
    GetOutput() = -1;
    return true;
  }

  int min_diff = std::numeric_limits<int>::max();
  int best_idx = -1;

  for (int i = 0; i < n - 1; ++i) {
    int diff = std::abs(v[i + 1] - v[i]);
    if (diff < min_diff) {
      min_diff = diff;
      best_idx = i;
    }
  }

  GetOutput() = best_idx;
  return true;
}

bool KuterginAClosestPairSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kutergin_a_closest_pair
