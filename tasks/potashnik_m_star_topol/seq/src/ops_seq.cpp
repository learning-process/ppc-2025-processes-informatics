#include "potashnik_m_star_topol/seq/include/ops_seq.hpp"

#include <string>

#include "potashnik_m_star_topol/common/include/common.hpp"

namespace potashnik_m_star_topol {

PotashnikMStarTopolSEQ::PotashnikMStarTopolSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PotashnikMStarTopolSEQ::ValidationImpl() {
  return GetInput().size() != 0;
}

bool PotashnikMStarTopolSEQ::PreProcessingImpl() {
  return true;
}

bool PotashnikMStarTopolSEQ::RunImpl() {
  // Sequential version just for CI tests to pass
  auto &input = GetInput();
  int size = input.size();

  int sum = 0;
  for (int i = 0; i < size; i++) {
    sum += input[i];
  }

  GetOutput() = sum;
  return true;
}

bool PotashnikMStarTopolSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace potashnik_m_star_topol
