#include "../include/ops_seq.hpp"

#include <cmath>
#include <numeric>
#include <vector>

#include "kruglova_a_max_diff_adjacent/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kruglova_a_max_diff_adjacent {

KruglovaAMaxDiffAdjacentSEQ::KruglovaAMaxDiffAdjacentSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0f;
}

bool KruglovaAMaxDiffAdjacentSEQ::ValidationImpl() {
  const auto &vec = GetInput();
  return vec.size() >= 2;
}

bool KruglovaAMaxDiffAdjacentSEQ::PreProcessingImpl() {
  GetOutput() = 0.0f;
  return true;
}

bool KruglovaAMaxDiffAdjacentSEQ::RunImpl() {
  const auto &vec = this->GetInput();
  auto &out = this->GetOutput();

  float max_diff = std::abs(vec[1] - vec[0]);
  for (size_t i = 1; i < vec.size(); ++i) {
    float diff = std::abs(vec[i] - vec[i - 1]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  out = max_diff;
  return true;
}

bool KruglovaAMaxDiffAdjacentSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_max_diff_adjacent
