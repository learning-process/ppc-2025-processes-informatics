#include "shvetsova_k_gausse_vert_strip/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
namespace shvetsova_k_gausse_vert_strip {

// Вспомогательная функция для определения ранга-владельца колонки k
int GetOwner(int k, int n, int size) {
  int base_cols = n / size;
  int remainder = n % size;
  int threshold = remainder * (base_cols + 1);
  if (k < threshold) {
    return k / (base_cols + 1);
  }
  return remainder + ((k - threshold) / base_cols);
}

// Вспомогательная функция для определения начального индекса колонки для ранга
int GetColStart(int rank, int n, int size) {
  int base_cols = n / size;
  int remainder = n % size;
  if (rank < remainder) {
    return rank * (base_cols + 1);
  }
  return (remainder * (base_cols + 1)) + ((rank - remainder) * base_cols);
}

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
  size_of_rib_ = 1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i != j && std::abs(matrix[i][j]) > 1e-10) {
        int dist = std::abs(i - j);
        size_of_rib_ = std::max(size_of_rib_, dist + 1);
      }
    }
  }
  return true;
}

void ShvetsovaKGaussVertStripMPI::ForwardStep(int k, int n, int local_cols, int col_start,
                                              std::vector<std::vector<double>> &a_local, std::vector<double> &b) const {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int owner = GetOwner(k, n, size);
  double pivot_val = 0.0;
  if (rank == owner) {
    pivot_val = a_local[k][k - col_start];
  }
  MPI_Bcast(&pivot_val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

  for (int j = 0; j < local_cols; j++) {
    a_local[k][j] /= pivot_val;
  }
  b[k] /= pivot_val;

  int last_row = std::min(n, k + size_of_rib_);
  for (int i = k + 1; i < last_row; i++) {
    double factor = 0.0;
    if (rank == owner) {
      factor = a_local[i][k - col_start];
    }
    MPI_Bcast(&factor, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    for (int j = 0; j < local_cols; j++) {
      a_local[i][j] -= factor * a_local[k][j];
    }
    b[i] -= factor * b[k];
  }
}

void ShvetsovaKGaussVertStripMPI::BackwardStep(int k, int n, int col_start, std::vector<std::vector<double>> &a_local,
                                               std::vector<double> &b, std::vector<double> &x) const{
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int owner = GetOwner(k, n, size);
  if (rank == owner) {
    x[k] = b[k];
  }
  MPI_Bcast(&x[k], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

  int first_row = std::max(0, k - size_of_rib_ + 1);
  for (int i = first_row; i < k; i++) {
    double val_ik = 0.0;
    if (rank == owner) {
      val_ik = a_local[i][k - col_start];
    }
    MPI_Bcast(&val_ik, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    b[i] -= val_ik * x[k];
  }
}

bool ShvetsovaKGaussVertStripMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &a_global = GetInput().first;
  int n = static_cast<int>(a_global.size());

  int col_start = GetColStart(rank, n, size);
  int next_col_start = GetColStart(rank + 1, n, size);
  int local_cols = next_col_start - col_start;

  std::vector<std::vector<double>> a_local(n, std::vector<double>(local_cols));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < local_cols; j++) {
      a_local[i][j] = a_global[i][col_start + j];
    }
  }
  std::vector<double> b = GetInput().second;

  for (int k = 0; k < n; k++) {
    ForwardStep(k, n, local_cols, col_start, a_local, b);
  }

  std::vector<double> x(n, 0.0);
  for (int k = n - 1; k >= 0; k--) {
    BackwardStep(k, n, col_start, a_local, b, x);
  }

  GetOutput().assign(x.begin(), x.end());
  return true;
}

bool ShvetsovaKGaussVertStripMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_gausse_vert_strip
