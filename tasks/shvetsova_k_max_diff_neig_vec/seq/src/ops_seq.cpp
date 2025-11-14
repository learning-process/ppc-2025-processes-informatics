#include "shvetsova_k_max_diff_neig_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_max_diff_neig_vec {

ShvetsovaKMaxDiffNeigVecSEQ::ShvetsovaKMaxDiffNeigVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::pair<double, std::pair<int, int>>{{0}, {0, 0}};
}

bool ShvetsovaKMaxDiffNeigVecSEQ::ValidationImpl() {
  data = GetInput();
  return true;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::PreProcessingImpl() {
  return true;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::RunImpl() {
  double MaxDif = 0;
  int FirstElem = 0;
  int SecondElem = 0;
  for (size_t i = 0; i < data.size() - 1; i++) {
    if (MaxDif <= abs(data.at(i) - data.at(i + 1))) {
      FirstElem = i;
      SecondElem = i + 1;
      MaxDif = abs(data.at(i) - data.at(i + 1));
    }
  }
  GetOutput().first = MaxDif;
  GetOutput().second.first = FirstElem;
  GetOutput().second.second = SecondElem;
  return true;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_max_diff_neig_vec
