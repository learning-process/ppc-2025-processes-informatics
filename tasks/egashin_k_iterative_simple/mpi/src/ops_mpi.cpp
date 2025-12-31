#include "egashin_k_iterative_simple/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "egashin_k_iterative_simple/common/include/common.hpp"

namespace egashin_k_iterative_simple {

EgashinKIterativeSimpleMPI::EgashinKIterativeSimpleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input = GetInput();
  input.A.clear();
  input.A.reserve(in.A.size());
  for (const auto &row : in.A) {
    input.A.push_back(row);
  }
  input.b.clear();
  input.b.reserve(in.b.size());
  input.b.insert(input.b.end(), in.b.begin(), in.b.end());
  input.x0.clear();
  input.x0.reserve(in.x0.size());
  input.x0.insert(input.x0.end(), in.x0.begin(), in.x0.end());
  input.tolerance = in.tolerance;
  input.max_iterations = in.max_iterations;
  GetOutput() = std::vector<double>(in.A.size(), 0.0);
}

bool EgashinKIterativeSimpleMPI::ValidationImpl() {
  const auto &input = GetInput();
  std::size_t n = input.A.size();

  if (n == 0) {
    return false;
  }

  for (std::size_t i = 0; i < n; ++i) {
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

bool EgashinKIterativeSimpleMPI::PreProcessingImpl() {
  return true;
}

void EgashinKIterativeSimpleMPI::CalculateDistribution(int size, int n, std::vector<int> &counts,
                                                       std::vector<int> &displs) {
  int delta = n / size;
  int remainder = n % size;
  for (int i = 0; i < size; ++i) {
    counts[i] = delta + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }
}

double EgashinKIterativeSimpleMPI::CalculateTau(const std::vector<std::vector<double>> &matrix, int start_row,
                                                int end_row) {
  double local_max_row_sum = 0.0;
  std::size_t n = matrix[0].size();

  for (int i = start_row; i < end_row; ++i) {
    double row_sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      row_sum += std::abs(matrix[i][j]);
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

double EgashinKIterativeSimpleMPI::CalculateNorm(const std::vector<double> &v) {
  double norm = 0.0;
  for (double val : v) {
    norm += val * val;
  }
  return std::sqrt(norm);
}

bool EgashinKIterativeSimpleMPI::CheckConvergence(const std::vector<double> &x_old, const std::vector<double> &x_new,
                                                  double tol) {
  double diff_norm = 0.0;
  for (std::size_t i = 0; i < x_old.size(); ++i) {
    double diff = x_new[i] - x_old[i];
    diff_norm += diff * diff;
  }
  return std::sqrt(diff_norm) < tol;
}

void EgashinKIterativeSimpleMPI::BroadcastInputData(int n, std::vector<std::vector<double>> &a_local,
                                                    std::vector<double> &b, std::vector<double> &x, double &tolerance,
                                                    int &max_iterations) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto &input = GetInput();

  if (rank == 0) {
    a_local = input.A;
    b = input.b;
    x = input.x0;
    tolerance = input.tolerance;
    max_iterations = input.max_iterations;
  }

  for (int i = 0; i < n; ++i) {
    MPI_Bcast(a_local[i].data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tolerance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void EgashinKIterativeSimpleMPI::PerformIteration(const std::vector<std::vector<double>> &a_local,
                                                  const std::vector<double> &b, const std::vector<double> &x,
                                                  std::vector<double> &x_new, int start_row, int local_rows, int n,
                                                  double tau, const std::vector<int> &counts,
                                                  const std::vector<int> &displs) {
  std::vector<double> local_x_new(local_rows);

  for (int i = 0; i < local_rows; ++i) {
    int global_i = start_row + i;
    double ax_i = 0.0;
    for (int j = 0; j < n; ++j) {
      ax_i += a_local[global_i][j] * x[j];
    }
    local_x_new[i] = x[global_i] + (tau * (b[global_i] - ax_i));
  }

  MPI_Allgatherv(local_x_new.data(), local_rows, MPI_DOUBLE, x_new.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);
}

bool EgashinKIterativeSimpleMPI::RunImpl() {
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
  int local_rows = counts[rank];

  std::vector<std::vector<double>> a_local(n, std::vector<double>(n));
  std::vector<double> b(n);
  std::vector<double> x(n);
  double tolerance = 0.0;
  int max_iterations = 0;

  BroadcastInputData(n, a_local, b, x, tolerance, max_iterations);

  double tau = CalculateTau(a_local, start_row, start_row + local_rows);

  std::vector<double> x_new(n);

  for (int iter = 0; iter < max_iterations; ++iter) {
    PerformIteration(a_local, b, x, x_new, start_row, local_rows, n, tau, counts, displs);

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

bool EgashinKIterativeSimpleMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egashin_k_iterative_simple
