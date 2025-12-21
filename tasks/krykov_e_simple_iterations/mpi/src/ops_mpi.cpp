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

void CalculateLocalXNew(int start, int count, size_t n, const std::vector<double> &a_local,
                        const std::vector<double> &b_local, const std::vector<double> &x,
                        std::vector<double> &local_x_new) {
  for (int i = 0; i < count; ++i) {
    int global_i = start + i;
    double sum = 0.0;

    const double *row = &a_local[i * n];
    for (size_t j = 0; j < n; ++j) {
      if (j != static_cast<size_t>(global_i)) {
        sum += row[j] * x[j];
      }
    }

    local_x_new[i] = (b_local[i] - sum) / row[global_i];
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

void CalculateCountsAndDispls(int size, int base, int rem, std::vector<int> &counts, std::vector<int> &displs) {
  if (counts.empty() || displs.empty() || size <= 0) {
    return;
  }

  counts.resize(size);
  displs.resize(size);

  counts[0] = base + (0 < rem ? 1 : 0);
  displs[0] = 0;
  for (int i = 1; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = displs[i - 1] + counts[i - 1];
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
  std::vector<double> a_global;
  std::vector<double> b_global;

  if (rank == 0) {
    const auto &input = GetInput();
    n = std::get<0>(input);
    a_global = std::get<1>(input);
    b_global = std::get<2>(input);
  }

  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  int base = static_cast<int>(n) / size;
  int rem = static_cast<int>(n) % size;

  int start = rank * base + std::min(rank, rem);
  int count = base + (rank < rem ? 1 : 0);

  std::vector<int> row_counts, row_displs;
  CalculateCountsAndDispls(size, base, rem, row_counts, row_displs);

  std::vector<int> a_counts(size), a_displs(size);
  for (int i = 0; i < size; ++i) {
    a_counts[i] = row_counts[i] * static_cast<int>(n);
    a_displs[i] = row_displs[i] * static_cast<int>(n);
  }

  std::vector<double> a_local(count * n);
  std::vector<double> b_local(count);

  MPI_Scatterv(a_global.data(), a_counts.data(), a_displs.data(), MPI_DOUBLE, a_local.data(),
               static_cast<int>(count * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(b_global.data(), row_counts.data(), row_displs.data(), MPI_DOUBLE, b_local.data(), count, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    CalculateLocalXNew(start, count, n, a_local, b_local, x, local_x_new);

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(), row_counts.data(), row_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

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
