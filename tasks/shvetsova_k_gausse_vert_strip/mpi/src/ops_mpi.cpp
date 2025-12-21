#include "shvetsova_k_gausse_vert_strip/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace shvetsova_k_gausse_vert_strip {

ShvetsovaKGaussVertStripMPI::ShvetsovaKGaussVertStripMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
}

bool ShvetsovaKGaussVertStripMPI::ValidationImpl() {
  return !GetInput().first.empty() && GetInput().first.size() == GetInput().second.size();
}

bool ShvetsovaKGaussVertStripMPI::PreProcessingImpl() {
  const auto &matrix = GetInput().first;
  int n = static_cast<int>(matrix.size());
  sizeOfRib = 1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i != j && std::abs(matrix[i][j]) > 1e-10) {
        int dist = std::abs(i - j);
        if (dist + 1 > sizeOfRib) {
          sizeOfRib = dist + 1;
        }
      }
    }
  }
  return true;
}

bool ShvetsovaKGaussVertStripMPI::RunImpl() {
  int size = 0, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &A_global = GetInput().first;
  const auto &b_global = GetInput().second;
  int N = static_cast<int>(A_global.size());

  // Распределение столбцов
  int base_cols = N / size;
  int remainder = N % size;
  int local_cols = (rank < remainder) ? (base_cols + 1) : base_cols;
  int col_start =
      (rank < remainder) ? rank * (base_cols + 1) : remainder * (base_cols + 1) + (rank - remainder) * base_cols;

  std::vector<std::vector<double>> A_local(N, std::vector<double>(local_cols));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < local_cols; j++) {
      A_local[i][j] = A_global[i][col_start + j];
    }
  }
  std::vector<double> b = b_global;

  // Прямой ход
  for (int k = 0; k < N; k++) {
    // Определяем владельца k-го столбца
    int owner = 0;
    int sum_cols = 0;
    for (int r = 0; r < size; r++) {
      int r_cols = (r < remainder) ? (base_cols + 1) : base_cols;
      if (k >= sum_cols && k < sum_cols + r_cols) {
        owner = r;
        break;
      }
      sum_cols += r_cols;
    }

    double pivot_val = 0.0;
    if (rank == owner) {
      pivot_val = A_local[k][k - col_start];
    }
    MPI_Bcast(&pivot_val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    // Нормализация текущей строки k (каждый нормализует свою часть)
    for (int j = 0; j < local_cols; j++) {
      A_local[k][j] /= pivot_val;
    }
    b[k] /= pivot_val;

    // Исключение элементов ниже k в пределах ленты
    int last_row = std::min(N, k + sizeOfRib);
    for (int i = k + 1; i < last_row; i++) {
      double factor = 0.0;
      if (rank == owner) {
        factor = A_local[i][k - col_start];
      }
      MPI_Bcast(&factor, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

      for (int j = 0; j < local_cols; j++) {
        A_local[i][j] -= factor * A_local[k][j];
      }
      b[i] -= factor * b[k];
    }
  }

  // Обратный ход
  std::vector<double> x(N, 0.0);
  for (int k = N - 1; k >= 0; k--) {
    int owner = 0;
    int sum_cols = 0;
    for (int r = 0; r < size; r++) {
      int r_cols = (r < remainder) ? (base_cols + 1) : base_cols;
      if (k >= sum_cols && k < sum_cols + r_cols) {
        owner = r;
        break;
      }
      sum_cols += r_cols;
    }

    if (rank == owner) {
      x[k] = b[k];
    }
    MPI_Bcast(&x[k], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    // Обновление векторов b для строк выше
    int first_row = std::max(0, k - sizeOfRib + 1);
    for (int i = first_row; i < k; i++) {
      double val_ik = 0.0;
      if (rank == owner) {
        val_ik = A_local[i][k - col_start];
      }
      MPI_Bcast(&val_ik, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
      b[i] -= val_ik * x[k];
    }
  }

  // Все процессы должны иметь результат
  GetOutput().assign(x.begin(), x.end());
  return true;
}

bool ShvetsovaKGaussVertStripMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_gausse_vert_strip
