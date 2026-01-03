#include "goriacheva_k_strassen_algorithm/seq/include/ops_seq.hpp"

#include <cmath>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"

namespace goriacheva_k_strassen_algorithm {

GoriachevaKStrassenAlgorithmSEQ::GoriachevaKStrassenAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GoriachevaKStrassenAlgorithmSEQ::ValidationImpl() {
  return is_square(GetInput().A) && is_square(GetInput().B) && GetInput().A.size() == GetInput().B.size();
}

bool GoriachevaKStrassenAlgorithmSEQ::PreProcessingImpl() {
  input_matrices_ = GetInput();
  return true;
}

bool GoriachevaKStrassenAlgorithmSEQ::RunImpl() {
  const auto &A = input_matrices_.A;
  const auto &B = input_matrices_.B;

  std::size_t n = A.size();
  std::size_t m = next_power_of_two(n);

  Matrix A_pad = (n == m) ? A : pad_matrix(A, m);
  Matrix B_pad = (n == m) ? B : pad_matrix(B, m);

  Matrix C_pad = strassen(A_pad, B_pad);
  result_matrix_ = (n == m) ? C_pad : crop_matrix(C_pad, n);

  return true;
}

bool GoriachevaKStrassenAlgorithmSEQ::PostProcessingImpl() {
  GetOutput() = result_matrix_;
  return true;
}

}  // namespace goriacheva_k_strassen_algorithm
