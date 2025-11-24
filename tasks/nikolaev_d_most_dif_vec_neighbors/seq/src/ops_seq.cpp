#include "nikolaev_d_most_dif_vec_neighbors/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"
#include "util/include/util.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

NikolaevDMostDifVecNeighborsSEQ::NikolaevDMostDifVecNeighborsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool NikolaevDMostDifVecNeighborsSEQ::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool NikolaevDMostDifVecNeighborsSEQ::PreProcessingImpl() {
  return true;
}

bool NikolaevDMostDifVecNeighborsSEQ::RunImpl() {
  if (GetInput().size() < 2) {
    return false;
  }

  auto &vec = GetInput();
  std::pair<int, int> result_elements;
  int max_diff = -1;

  for (size_t i = 0; i < GetInput().size() - 1; i++) {
    int j = i + 1;
    int diff = std::abs(vec[j] - vec[i]);
    if (diff > max_diff) {
      max_diff = diff;
      result_elements = {vec[i], vec[j]};
    }
  }

  GetOutput() = result_elements;
  return true;
}

bool NikolaevDMostDifVecNeighborsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikolaev_d_most_dif_vec_neighbors
