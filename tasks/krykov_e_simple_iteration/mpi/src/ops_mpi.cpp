#include "krykov_e_simple_iteration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "krykov_e_simple_iteration/common/include/common.hpp"

namespace krykov_e_simple_iteration {

KrykovESimpleIterationMPI::KrykovESimpleIterationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationMPI::ValidationImpl() {
  const auto &[n, A, b] = GetInput();
  return n > 0 && A.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationMPI::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationMPI::RunImpl() {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &[n, A, b] = GetInput();

  int base = n / size;
  int rem = n % size;
  int start = rank * base + std::min(rank, rem);
  int count = base + (rank < rem ? 1 : 0);

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  constexpr double eps = 1e-5;
  constexpr int max_iter = 10000;

  std::vector<int> recv_counts(size), displs(size);
  for (int i = 0; i < size; ++i) {
    recv_counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + recv_counts[i - 1];
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    for (int i = 0; i < count; ++i) {
      int global_i = start + i;
      double sum = 0.0;

      for (size_t j = 0; j < n; ++j) {
        if (j != static_cast<size_t>(global_i)) {
          sum += A[global_i * n + j] * x[j];
        }
      }
      local_x_new[i] = (b[global_i] - sum) / A[global_i * n + global_i];
    }

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double local_norm = 0.0;
    for (int i = 0; i < count; ++i) {
      int gi = start + i;
      double diff = x_new[gi] - x[gi];
      local_norm += diff * diff;
    }

    double global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x = x_new;
    if (std::sqrt(global_norm) < eps) {
      break;
    }
  }

  GetOutput() = x;

  return true;
}

bool KrykovESimpleIterationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_simple_iteration
