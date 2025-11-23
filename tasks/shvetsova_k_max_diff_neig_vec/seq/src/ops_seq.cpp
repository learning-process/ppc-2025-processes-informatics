#include "shvetsova_k_max_diff_neig_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_max_diff_neig_vec {

ShvetsovaKMaxDiffNeigVecSEQ::ShvetsovaKMaxDiffNeigVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::pair<double, double>{0.0, 0.0};
}

bool ShvetsovaKMaxDiffNeigVecSEQ::ValidationImpl() {
  data = GetInput();
  return data.size() >= 0;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::PreProcessingImpl() {
  return true;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::RunImpl() {
  double MaxDif = 0;
  double FirstElem = 0;
  double SecondElem = 0;
  int sz = data.size();
  if (sz < 2) {
    GetOutput().first = 0.0;
    GetOutput().second = 0.0;
    return true;
  }
  for (int i = 0; i < sz - 1; i++) {
    if (MaxDif <= std::abs(data.at(i) - data.at(i + 1))) {
      FirstElem = data.at(i);
      SecondElem = data.at(i + 1);
      MaxDif = std::abs(data.at(i) - data.at(i + 1));
    }
  }

  GetOutput().first = FirstElem;
  GetOutput().second = SecondElem;
  return GetOutput().first == FirstElem && GetOutput().second == SecondElem;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_max_diff_neig_vec
