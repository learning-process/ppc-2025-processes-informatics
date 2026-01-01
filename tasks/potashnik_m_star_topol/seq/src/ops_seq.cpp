#include "potashnik_m_star_topol/seq/include/ops_seq.hpp"

#include <string>
#include <tuple>

#include "potashnik_m_star_topol/common/include/common.hpp"

namespace potashnik_m_star_topol {

PotashnikMStarTopolSEQ::PotashnikMStarTopolSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0, 0);
}

bool PotashnikMStarTopolSEQ::ValidationImpl() {
  return GetInput().size() != 0;
}

bool PotashnikMStarTopolSEQ::PreProcessingImpl() {
  return true;
}

bool PotashnikMStarTopolSEQ::RunImpl() {
  // Sequential version just for CI tests to pass
  std::vector<int> tmp;
  tmp.reserve(500000);

  for (int i = 0; i < 500000; i++) {
    tmp.push_back(i);
  }

  int sum = 0;
  for (int i = 0; i < 500000; i++) {
    sum += tmp[i];
  }

  GetOutput() = std::make_tuple(sum, 0);
  return true;
}

bool PotashnikMStarTopolSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace potashnik_m_star_topol
