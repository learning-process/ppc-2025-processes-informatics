#include "votincev_d_matrix_mult/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>
#include <tuple>

#include "votincev_d_matrix_mult/common/include/common.hpp"

namespace votincev_d_matrix_mult {

VotincevDMatrixMultSEQ::VotincevDMatrixMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VotincevDMatrixMultSEQ::ValidationImpl() {
  const auto &in = GetInput();
  int m = std::get<0>(in);
  int n = std::get<1>(in);
  int k = std::get<2>(in);
  const auto &A = std::get<3>(in);
  const auto &B = std::get<4>(in);

  return (m > 0 && n > 0 && k > 0 &&
          A.size() == static_cast<size_t>(m * k) &&
          B.size() == static_cast<size_t>(k * n));
}

bool VotincevDMatrixMultSEQ::PreProcessingImpl() {
  auto &in = GetInput();
  m_ = std::get<0>(in);
  n_ = std::get<1>(in);
  k_ = std::get<2>(in);
  A_ = std::get<3>(in);
  B_ = std::get<4>(in);

  result_.assign(m_ * n_, 0.0);
  return true;
}

bool VotincevDMatrixMultSEQ::RunImpl() {
  for (int i = 0; i < m_; ++i) {
    for (int j = 0; j < n_; ++j) {
      double sum = 0.0;
      for (int t = 0; t < k_; ++t) {
        sum += A_[i * k_ + t] * B_[t * n_ + j];
      }
      result_[i * n_ + j] = sum;
    }
  }
  return true;
}

bool VotincevDMatrixMultSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace votincev_d_matrix_mult
