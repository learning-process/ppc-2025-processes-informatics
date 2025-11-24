#include "shvetsova_k_max_diff_neig_vec/seq/include/ops_seq.hpp"

// #include <cmath>

#include "shvetsova_k_max_diff_neig_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_max_diff_neig_vec {

ShvetsovaKMaxDiffNeigVecSEQ::ShvetsovaKMaxDiffNeigVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::pair<double, double>{0.0, 0.0};
}

bool ShvetsovaKMaxDiffNeigVecSEQ::ValidationImpl() {
  data_ = GetInput();
  return true;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::PreProcessingImpl() {
  return true;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::RunImpl() {
  double max_dif = 0;
  double first_elem = 0;
  double second_elem = 0;
  int sz = data_.size();
  if (sz < 2) {
    GetOutput().first = 0.0;
    GetOutput().second = 0.0;
    return true;
  }
  for (int i = 0; i < sz - 1; i++) {
    if (max_dif <= std::abs(data_.at(i) - data_.at(i + 1))) {
      first_elem = data_.at(i);
      second_elem = data_.at(i + 1);
      max_dif = std::abs(data_.at(i) - data_.at(i + 1));
    }
  }

  GetOutput().first = first_elem;
  GetOutput().second = second_elem;
  return true;
}

bool ShvetsovaKMaxDiffNeigVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_max_diff_neig_vec
