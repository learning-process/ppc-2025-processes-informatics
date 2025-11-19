#include "kutergin_a_closest_pair/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "kutergin_a_closest_pair/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kutergin_a_closest_pair {

KuterginAClosestPairSEQ::KuterginAClosestPairSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = -1;
}

bool KuterginAClosestPairSEQ::ValidationImpl() {
  return true;
}

bool KuterginAClosestPairSEQ::PreProcessingImpl() {
  return true;
}

bool KuterginAClosestPairSEQ::RunImpl() {
  const auto &v = GetInput();
  if (v.size() < 2) {
    GetOutput() = -1;
    return true;
  }

  int best_idx = 0;
  int min_diff = std::abs(v[1] - v[0]);

  for (size_t i = 1; i < v.size() - 1; ++i) {
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
