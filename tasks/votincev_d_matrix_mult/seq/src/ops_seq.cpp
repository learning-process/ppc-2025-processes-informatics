#include "votincev_d_matrix_mult/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

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

  return (m > 0 && n > 0 && k > 0 && A.size() == static_cast<size_t>(m * k) && B.size() == static_cast<size_t>(k * n));
}

bool VotincevDMatrixMultSEQ::PreProcessingImpl() {
  return true;
}

bool VotincevDMatrixMultSEQ::RunImpl() {
  int param_m = 0, param_n = 0, param_k = 0;
  std::vector<double> matrix_A;
  std::vector<double> matrix_B;
  std::vector<double> matrix_res;

  auto &in = GetInput();
  param_m = std::get<0>(in);
  param_n = std::get<1>(in);
  param_k = std::get<2>(in);
  matrix_A = std::get<3>(in);
  matrix_B = std::get<4>(in);

  matrix_res.assign(param_m * param_n, 0.0);

  for (int i = 0; i < param_m; ++i) {
    for (int j = 0; j < param_n; ++j) {
      double sum = 0.0;
      for (int t = 0; t < param_k; ++t) {
        sum += matrix_A[i * param_k + t] * matrix_B[t * param_n + j];
      }
      matrix_res[i * param_n + j] = sum;
    }
  }

  GetOutput() = matrix_res;
  return true;
}

bool VotincevDMatrixMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_matrix_mult
