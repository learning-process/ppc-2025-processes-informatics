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
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  const auto &[n, A, b] = GetInput();

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new_local(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  constexpr double eps = 1e-5;
  constexpr int max_iter = 100;

  int base = n / world_size;
  int rem = n % world_size;

  int start = world_rank * base + std::min<int>(world_rank, rem);
  int count = base + (world_rank < rem ? 1 : 0);
  int end = start + count;

  for (int iter = 0; iter < max_iter; ++iter) {
    std::fill(x_new_local.begin(), x_new_local.end(), 0.0);

    // локальное вычисление
    for (int i = start; i < end; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          sum += A[i * n + j] * x[j];
        }
      }
      x_new_local[i] = (b[i] - sum) / A[i * n + i];
    }

    // сборка полного вектора
    MPI_Allreduce(x_new_local.data(), x_new.data(), static_cast<int>(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // вычисление нормы
    double local_diff = 0.0;
    for (int i = start; i < end; ++i) {
      local_diff = std::max(local_diff, std::abs(x_new[i] - x[i]));
    }

    double global_diff = 0.0;
    MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    x = x_new;
    if (global_diff < eps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool KrykovESimpleIterationMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace krykov_e_simple_iteration
