#include "krykov_e_simple_iterations/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

#include "krykov_e_simple_iterations/common/include/common.hpp"

namespace krykov_e_simple_iterations {
namespace {

constexpr double kEps = 1e-5;
constexpr int kMaxIter = 10000;

void CalculateLocalXNew(int start, int count, size_t n, const std::vector<double> &a, const std::vector<double> &b,
                        const std::vector<double> &x, std::vector<double> &local_x_new) {
  for (int i = 0; i < count; ++i) {
    int global_i = start + i;
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      if (std::cmp_not_equal(j, global_i)) {
        sum += a[(global_i * n) + j] * x[j];
      }
    }
    local_x_new[i] = (b[global_i] - sum) / a[(global_i * n) + global_i];
  }
}

double CalculateLocalNorm(int start, int count, const std::vector<double> &x_new, const std::vector<double> &x) {
  double local_norm = 0.0;
  for (int i = 0; i < count; ++i) {
    int gi = start + i;
    double diff = x_new[gi] - x[gi];
    local_norm += diff * diff;
  }
  return local_norm;
}

void CalculateRecvCountsAndDispls(int size, int base, int rem, std::vector<int> &recv_counts,
                                  std::vector<int> &displs) {
  if (recv_counts.empty() || displs.empty() || size <= 0) {
    return;
  }

  recv_counts[0] = base + (0 < rem ? 1 : 0);
  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    recv_counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }
}

}  // namespace

KrykovESimpleIterationsMPI::KrykovESimpleIterationsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationsMPI::ValidationImpl() {
  const auto &[n, a, b] = GetInput();
  return n > 0 && a.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationsMPI::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationsMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t n = 0;
  std::vector<double> a;
  std::vector<double> b;

  if (rank == 0) {
    const auto &input = GetInput();
    n = std::get<0>(input);
    a = std::get<1>(input);
    b = std::get<2>(input);
  }

  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    a.resize(n * n);
    b.resize(n);
  }

  MPI_Bcast(a.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b.data(), static_cast<int>(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int base = static_cast<int>(n) / size;
  int rem = static_cast<int>(n) % size;
  int start = (rank * base) + std::min(rank, rem);
  int count = base + (rank < rem ? 1 : 0);

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  CalculateRecvCountsAndDispls(size, base, rem, recv_counts, displs);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    CalculateLocalXNew(start, count, n, a, b, x, local_x_new);

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double local_norm = CalculateLocalNorm(start, count, x_new, x);
    double global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x = x_new;
    if (std::sqrt(global_norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool KrykovESimpleIterationsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_simple_iterations
