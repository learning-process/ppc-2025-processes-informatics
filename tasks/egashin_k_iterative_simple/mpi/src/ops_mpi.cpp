#include "egashin_k_iterative_simple/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace egashin_k_iterative_simple {

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>(in.A.size(), 0.0);
}

bool TestTaskMPI::ValidationImpl() {
  const auto &input = GetInput();
  size_t n = input.A.size();
  
  if (n == 0) {
    return false;
  }
  
  for (size_t i = 0; i < n; ++i) {
    if (input.A[i].size() != n) {
      return false;
    }
  }
  
  if (input.b.size() != n || input.x0.size() != n) {
    return false;
  }
  
  if (input.tolerance <= 0 || input.max_iterations <= 0) {
    return false;
  }
  
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  return true;
}

void TestTaskMPI::CalculateDistribution(int size, int n, std::vector<int> &counts, std::vector<int> &displs) {
  int delta = n / size;
  int remainder = n % size;
  for (int i = 0; i < size; ++i) {
    counts[i] = delta + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }
}

double TestTaskMPI::CalculateTau(const std::vector<std::vector<double>> &A, int start_row, int end_row) {
  double local_max_row_sum = 0.0;
  size_t n = A[0].size();
  
  for (int i = start_row; i < end_row; ++i) {
    double row_sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      row_sum += std::abs(A[i][j]);
    }
    local_max_row_sum = std::max(local_max_row_sum, row_sum);
  }
  
  double global_max_row_sum = 0.0;
  MPI_Allreduce(&local_max_row_sum, &global_max_row_sum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  
  if (global_max_row_sum < 1e-10) {
    return 0.1;
  }
  return 1.0 / (global_max_row_sum + 1.0);
}

double TestTaskMPI::CalculateNorm(const std::vector<double> &v) {
  double norm = 0.0;
  for (double val : v) {
    norm += val * val;
  }
  return std::sqrt(norm);
}

bool TestTaskMPI::CheckConvergence(const std::vector<double> &x_old, const std::vector<double> &x_new, double tolerance) {
  double diff_norm = 0.0;
  for (size_t i = 0; i < x_old.size(); ++i) {
    double diff = x_new[i] - x_old[i];
    diff_norm += diff * diff;
  }
  return std::sqrt(diff_norm) < tolerance;
}

bool TestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  int n = static_cast<int>(input.A.size());

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  if (rank == 0) {
    CalculateDistribution(size, n, counts, displs);
  }
  MPI_Bcast(counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  int start_row = displs[rank];
  int end_row = displs[rank] + counts[rank];
  int local_rows = counts[rank];

  std::vector<std::vector<double>> A_local(n, std::vector<double>(n));
  if (rank == 0) {
    A_local = input.A;
  }
  
  for (int i = 0; i < n; ++i) {
    MPI_Bcast(A_local[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  std::vector<double> b(n);
  if (rank == 0) {
    b = input.b;
  }
  MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(n);
  if (rank == 0) {
    x = input.x0;
  }
  MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double tolerance = input.tolerance;
  int max_iterations = input.max_iterations;
  MPI_Bcast(&tolerance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

  double tau = CalculateTau(A_local, start_row, end_row);

  std::vector<double> x_new(n);
  std::vector<double> local_x_new(local_rows);

  for (int iter = 0; iter < max_iterations; ++iter) {
    for (int i = 0; i < local_rows; ++i) {
      int global_i = start_row + i;
      double Ax_i = 0.0;
      for (int j = 0; j < n; ++j) {
        Ax_i += A_local[global_i][j] * x[j];
      }
      local_x_new[i] = x[global_i] + tau * (b[global_i] - Ax_i);
    }

    MPI_Allgatherv(local_x_new.data(), local_rows, MPI_DOUBLE, x_new.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    bool converged = false;
    if (rank == 0) {
      converged = CheckConvergence(x, x_new, tolerance);
    }
    int converged_int = converged ? 1 : 0;
    MPI_Bcast(&converged_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    converged = (converged_int == 1);

    if (converged) {
      break;
    }

    x = x_new;
  }

  if (rank == 0) {
    GetOutput() = x_new;
  }

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egashin_k_iterative_simple

