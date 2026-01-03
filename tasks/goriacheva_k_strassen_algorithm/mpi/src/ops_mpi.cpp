#include "goriacheva_k_strassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "goriacheva_k_strassen_algorithm/common/include/common.hpp"

namespace goriacheva_k_strassen_algorithm {

GoriachevaKStrassenAlgorithmMPI::GoriachevaKStrassenAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool GoriachevaKStrassenAlgorithmMPI::ValidationImpl() {
  return is_square(GetInput().A) && is_square(GetInput().B) && GetInput().A.size() == GetInput().B.size();
}

bool GoriachevaKStrassenAlgorithmMPI::PreProcessingImpl() {
  input_matrices_ = GetInput();
  return true;
}

Matrix GoriachevaKStrassenAlgorithmMPI::mpi_strassen_top(const Matrix &A, const Matrix &B) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::size_t n = A.size();
  if (size == 1 || n <= 1) {
    return strassen(A, B);
  }

  std::size_t k = n / 2;

  Matrix A11(k, std::vector<double>(k)), A12(k, std::vector<double>(k));
  Matrix A21(k, std::vector<double>(k)), A22(k, std::vector<double>(k));
  Matrix B11(k, std::vector<double>(k)), B12(k, std::vector<double>(k));
  Matrix B21(k, std::vector<double>(k)), B22(k, std::vector<double>(k));

  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      A11[i][j] = A[i][j];
      A12[i][j] = A[i][j + k];
      A21[i][j] = A[i + k][j];
      A22[i][j] = A[i + k][j + k];
      B11[i][j] = B[i][j];
      B12[i][j] = B[i][j + k];
      B21[i][j] = B[i + k][j];
      B22[i][j] = B[i + k][j + k];
    }
  }

  int num_tasks = std::min(7, size);
  int task_id = rank % num_tasks;

  MPI_Comm subcomm;
  MPI_Comm_split(MPI_COMM_WORLD, task_id, rank, &subcomm);

  int sub_rank;
  MPI_Comm_rank(subcomm, &sub_rank);

  Matrix Mi;
  if (sub_rank == 0) {
    switch (task_id) {
      case 0:
        Mi = strassen(add(A11, A22), add(B11, B22));
        break;
      case 1:
        Mi = strassen(add(A21, A22), B11);
        break;
      case 2:
        Mi = strassen(A11, sub(B12, B22));
        break;
      case 3:
        Mi = strassen(A22, sub(B21, B11));
        break;
      case 4:
        Mi = strassen(add(A11, A12), B22);
        break;
      case 5:
        Mi = strassen(sub(A21, A11), add(B11, B12));
        break;
      case 6:
        Mi = strassen(sub(A12, A22), add(B21, B22));
        break;
    }
  }

  MPI_Comm_free(&subcomm);

  std::vector<Matrix> M(7);

  if (rank == 0) {
    if (task_id < num_tasks && sub_rank == 0) {
      M[task_id] = Mi;
    }

    for (int i = 1; i < num_tasks; ++i) {
      int tid;
      std::vector<double> buf(k * k);
      MPI_Status status;

      MPI_Recv(&tid, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(buf.data(), k * k, MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      M[tid] = unflatten(buf, k);
    }

    for (int t = num_tasks; t < 7; ++t) {
      switch (t) {
        case 0:
          M[0] = strassen(add(A11, A22), add(B11, B22));
          break;
        case 1:
          M[1] = strassen(add(A21, A22), B11);
          break;
        case 2:
          M[2] = strassen(A11, sub(B12, B22));
          break;
        case 3:
          M[3] = strassen(A22, sub(B21, B11));
          break;
        case 4:
          M[4] = strassen(add(A11, A12), B22);
          break;
        case 5:
          M[5] = strassen(sub(A21, A11), add(B11, B12));
          break;
        case 6:
          M[6] = strassen(sub(A12, A22), add(B21, B22));
          break;
      }
    }
  } else if (sub_rank == 0) {
    auto buf = flatten(Mi);
    MPI_Send(&task_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(buf.data(), k * k, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }

  Matrix C(n, std::vector<double>(n));
  if (rank == 0) {
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        C[i][j] = M[0][i][j] + M[3][i][j] - M[4][i][j] + M[6][i][j];
        C[i][j + k] = M[2][i][j] + M[4][i][j];
        C[i + k][j] = M[1][i][j] + M[3][i][j];
        C[i + k][j + k] = M[0][i][j] - M[1][i][j] + M[2][i][j] + M[5][i][j];
      }
    }
  }

  auto flatC = flatten(C);
  MPI_Bcast(flatC.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    C = unflatten(flatC, n);
  }

  return C;
}

bool GoriachevaKStrassenAlgorithmMPI::RunImpl() {
  const auto &A = input_matrices_.A;
  const auto &B = input_matrices_.B;

  std::size_t n = A.size();
  std::size_t m = next_power_of_two(n);

  Matrix A_pad = (n == m) ? A : pad_matrix(A, m);
  Matrix B_pad = (n == m) ? B : pad_matrix(B, m);

  Matrix C_pad = mpi_strassen_top(A_pad, B_pad);
  result_matrix_ = (n == m) ? C_pad : crop_matrix(C_pad, n);

  return true;
}

bool GoriachevaKStrassenAlgorithmMPI::PostProcessingImpl() {
  GetOutput() = result_matrix_;
  return true;
}

}  // namespace goriacheva_k_strassen_algorithm
