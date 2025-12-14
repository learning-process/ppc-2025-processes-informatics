#include "krykov_e_simple_iteration/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include "krykov_e_simple_iteration/common/include/common.hpp"

namespace krykov_e_simple_iteration {
namespace {

constexpr double kEps = 1e-5;
constexpr int kMaxIter = 10000;

void CalculateLocalXNew(int start, int count, size_t n,
                       const std::vector<double>& A,
                       const std::vector<double>& b,
                       const std::vector<double>& x,
                       std::vector<double>* local_x_new) {
  for (int i = 0; i < count; ++i) {
    int global_i = start + i;
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      if (j != static_cast<size_t>(global_i)) {
        sum += A[(global_i * n) + j] * x[j];
      }
    }
    (*local_x_new)[i] = (b[global_i] - sum) / A[(global_i * n) + global_i];
  }
}

double CalculateLocalNorm(int start, int count,
                         const std::vector<double>& x_new,
                         const std::vector<double>& x) {
  double local_norm = 0.0;
  for (int i = 0; i < count; ++i) {
    int gi = start + i;
    double diff = x_new[gi] - x[gi];
    local_norm += diff * diff;
  }
  return local_norm;
}

void CalculateRecvCountsAndDispls(int size, int base, int rem,
                                 std::vector<int>* recv_counts,
                                 std::vector<int>* displs) {
  (*recv_counts)[0] = base + (0 < rem ? 1 : 0);
  (*displs)[0] = 0;
  for (int i = 1; i < size; ++i) {
    (*recv_counts)[i] = base + (i < rem ? 1 : 0);
    (*displs)[i] = (*displs)[i - 1] + (*recv_counts)[i - 1];
  }
}

}  // namespace

KrykovESimpleIterationMPI::KrykovESimpleIterationMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationMPI::ValidationImpl() {
  const auto& [n, A, b] = GetInput();
  return n > 0 && A.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationMPI::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto& [n, A, b] = GetInput();

  int base = static_cast<int>(n) / size;
  int rem = static_cast<int>(n) % size;
  int start = (rank * base) + std::min(rank, rem);
  int count = base + (rank < rem ? 1 : 0);

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  CalculateRecvCountsAndDispls(size, base, rem, &recv_counts, &displs);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    CalculateLocalXNew(start, count, n, A, b, x, &local_x_new);

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(),
                   recv_counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double local_norm = CalculateLocalNorm(start, count, x_new, x);
    double global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    x = x_new;
    if (std::sqrt(global_norm) < kEps) {
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