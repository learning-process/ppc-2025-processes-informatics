#include "goriacheva_k_strassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <vector>
#include <algorithm>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"

namespace goriacheva_k_strassen_algorithm {

GoriachevaKStrassenAlgorithmMPI::GoriachevaKStrassenAlgorithmMPI(
    const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GoriachevaKStrassenAlgorithmMPI::ValidationImpl() {
  return is_square(GetInput().A) &&
         is_square(GetInput().B) &&
         GetInput().A.size() == GetInput().B.size();
}

bool GoriachevaKStrassenAlgorithmMPI::PreProcessingImpl() {
  input_matrices_ = GetInput();
  return true;
}

bool GoriachevaKStrassenAlgorithmMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const Matrix& A = input_matrices_.A;
  const Matrix& B = input_matrices_.B;
  const size_t n = A.size();

  if (n == 1) {
    result_matrix_.assign(1, std::vector<double>(1));

    if (rank == 0) {
      result_matrix_[0][0] = A[0][0] * B[0][0];
    }

    return true;
  }


  // ===============================
  // 1. ближайшая степень двойки
  // ===============================
  size_t m = 1;
  if (rank == 0) {
    while (m < n) m <<= 1;
  }
  MPI_Bcast(&m, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // ===============================
  // 2. padding матриц
  // ===============================
  Matrix Ap(m, std::vector<double>(m, 0.0));
  Matrix Bp(m, std::vector<double>(m, 0.0));

  if (rank == 0) {
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j) {
        Ap[i][j] = A[i][j];
        Bp[i][j] = B[i][j];
      }
  }

  // ===============================
  // 3. broadcast padded matrices
  // ===============================
  for (size_t i = 0; i < m; ++i) {
    MPI_Bcast(Ap[i].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Bp[i].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  // ===============================
  // 4. split matrices
  // ===============================
  const size_t k = m / 2;

  auto split = [&](const Matrix& M, size_t r, size_t c) {
    Matrix out(k, std::vector<double>(k));
    for (size_t i = 0; i < k; ++i)
      for (size_t j = 0; j < k; ++j)
        out[i][j] = M[i + r * k][j + c * k];
    return out;
  };

  Matrix A11 = split(Ap, 0, 0);
  Matrix A12 = split(Ap, 0, 1);
  Matrix A21 = split(Ap, 1, 0);
  Matrix A22 = split(Ap, 1, 1);

  Matrix B11 = split(Bp, 0, 0);
  Matrix B12 = split(Bp, 0, 1);
  Matrix B21 = split(Bp, 1, 0);
  Matrix B22 = split(Bp, 1, 1);

  // ===============================
  // 5. инициализация M1–M7
  // ===============================
  Matrix zero(k, std::vector<double>(k, 0.0));
  Matrix M1 = zero, M2 = zero, M3 = zero, M4 = zero;
  Matrix M5 = zero, M6 = zero, M7 = zero;

  // ===============================
  // 6. распределение задач
  // ===============================
  int task = rank % 7;

  if (task == 0)
    M1 = naive_multiply(add(A11, A22), add(B11, B22));
  if (task == 1)
    M2 = naive_multiply(add(A21, A22), B11);
  if (task == 2)
    M3 = naive_multiply(A11, sub(B12, B22));
  if (task == 3)
    M4 = naive_multiply(A22, sub(B21, B11));
  if (task == 4)
    M5 = naive_multiply(add(A11, A12), B22);
  if (task == 5)
    M6 = naive_multiply(sub(A21, A11), add(B11, B12));
  if (task == 6)
    M7 = naive_multiply(sub(A12, A22), add(B21, B22));

// ===============================
  // 7. корректный MPI_Reduce
  // ===============================
  auto flatten = [&](const Matrix& M) {
    std::vector<double> v(k * k);
    for (size_t i = 0; i < k; ++i)
      for (size_t j = 0; j < k; ++j)
        v[i * k + j] = M[i][j];
    return v;
  };

  auto unflatten = [&](const std::vector<double>& v) {
    Matrix M(k, std::vector<double>(k));
    for (size_t i = 0; i < k; ++i)
      for (size_t j = 0; j < k; ++j)
        M[i][j] = v[i * k + j];
    return M;
  };

  auto reduce_matrix = [&](Matrix& M) {
    std::vector<double> send = flatten(M);
    std::vector<double> recv(k * k, 0.0);

    MPI_Reduce(send.data(), recv.data(),
               static_cast<int>(k * k),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
      M = unflatten(recv);
  };

  reduce_matrix(M1);
  reduce_matrix(M2);
  reduce_matrix(M3);
  reduce_matrix(M4);
  reduce_matrix(M5);
  reduce_matrix(M6);
  reduce_matrix(M7);

  // ===============================
  // 8. сборка результата
  // ===============================
  Matrix Cp(m, std::vector<double>(m, 0.0));

  if (rank == 0) {
    Matrix C11 = add(sub(add(M1, M4), M5), M7);
    Matrix C12 = add(M3, M5);
    Matrix C21 = add(M2, M4);
    Matrix C22 = add(sub(add(M1, M3), M2), M6);

    for (size_t i = 0; i < k; ++i)
      for (size_t j = 0; j < k; ++j) {
        Cp[i][j] = C11[i][j];
        Cp[i][j + k] = C12[i][j];
        Cp[i + k][j] = C21[i][j];
        Cp[i + k][j + k] = C22[i][j];
      }
  }

  // ===============================
  // 9. broadcast результата
  // ===============================
  for (size_t i = 0; i < m; ++i)
    MPI_Bcast(Cp[i].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // ===============================
  // 10. обрезка
  // ===============================
  result_matrix_.assign(n, std::vector<double>(n));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      result_matrix_[i][j] = Cp[i][j];

  return true;
}

bool GoriachevaKStrassenAlgorithmMPI::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = result_matrix_;
  }else{
    GetOutput().clear();
  }

  return true;
}

}  // namespace goriacheva_k_strassen_algorithm
