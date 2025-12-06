#include "lukin_i_cannon_algorithm/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "lukin_i_cannon_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace lukin_i_cannon_algorithm {

LukinICannonAlgorithmSEQ::LukinICannonAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LukinICannonAlgorithmSEQ::ValidationImpl() {
  int rsizeA = std::get<0>(GetInput()).size();
  int rsizeB = std::get<1>(GetInput()).size();
  int size = std::get<2>(GetInput());
  return (rsizeA > 0) && (rsizeB > 0) && (sqrt(rsizeA) == size) && (rsizeA == rsizeB);
}

bool LukinICannonAlgorithmSEQ::PreProcessingImpl() {
  return true;
}

bool LukinICannonAlgorithmSEQ::RunImpl() {
  double *A = std::get<0>(GetInput()).data();
  double *B = std::get<1>(GetInput()).data();
  size_ = std::get<2>(GetInput());

  std::vector<double> C(size_ * size_, 0);
  double *Cdata = C.data();

  for (int i = 0; i < size_; i++) {
    for (int k = 0; k < size_; k++) {
      double fixed = A[i * size_ + k];
      for (int j = 0; j < size_; j++) {
        Cdata[i * size_ + j] += fixed * B[k * size_ + j];
      }
    }
  }

  GetOutput() = std::move(C);
  return true;
}

bool LukinICannonAlgorithmSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_cannon_algorithm
