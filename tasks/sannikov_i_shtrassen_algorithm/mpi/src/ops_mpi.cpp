#include "sannikov_i_shtrassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <tuple>
#include <vector>

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"

namespace sannikov_i_shtrassen_algorithm {

SannikovIShtrassenAlgorithmMPI::SannikovIShtrassenAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

//
bool SannikovIShtrassenAlgorithmMPI::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);

  if (a.empty() || b.empty()) {
    return false;
  }
  if (a.size() != b.size()) {
    return false;
  }
  if (a.front().empty() || b.front().empty()) {
    return false;
  }

  const std::size_t n = a.size();
  for (const auto &row : a) {
    if (row.size() != n) {
      return false;
    }
  }
  for (const auto &row : b) {
    if (row.size() != n) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

using Matrix = std::vector<std::vector<double>>;

std::size_t NextPow2(std::size_t v) {
  std::size_t p = 1;
  while (p < v) {
    p <<= 1U;
  }
  return p;
}

bool CheckSizeIsOk(std::uint64_t n64) {
  if (n64 == 0) {
    return false;
  }
  if (n64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  return true;
}

void BuildRowPartition(int size, int rows, int cols, std::vector<int> *counts, std::vector<int> *displs) {
  counts->assign(size, 0);
  displs->assign(size, 0);

  const int base = rows / size;
  const int rem = rows % size;

  int disp = 0;
  for (int r = 0; r < size; ++r) {
    const int r_rows = base + ((r < rem) ? 1 : 0);
    (*counts)[r] = r_rows * cols;
    (*displs)[r] = disp;
    disp += (*counts)[r];
  }
}

void PackPaddedFlatOnRoot(const Matrix &src, int n0, int m, std::vector<double> *flat) {
  flat->assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n0; ++j) {
      (*flat)[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] =
          src[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }
}

Matrix UnpackCropFromFlat(const std::vector<double> &flat, int m, int n0) {
  Matrix out(static_cast<std::size_t>(n0), std::vector<double>(static_cast<std::size_t>(n0), 0.0));
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n0; ++j) {
      out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)];
    }
  }
  return out;
}

void MultiplyRowsClassic(const std::vector<double> &a_local, int r_rows, const std::vector<double> &b_full, int m,
                         std::vector<double> *c_local) {
  c_local->assign(static_cast<std::size_t>(r_rows) * static_cast<std::size_t>(m), 0.0);

  for (int i = 0; i < r_rows; ++i) {
    const double *arow = &a_local[static_cast<std::size_t>(i) * static_cast<std::size_t>(m)];
    double *crow = &(*c_local)[static_cast<std::size_t>(i) * static_cast<std::size_t>(m)];
    for (int k = 0; k < m; ++k) {
      const double aik = arow[static_cast<std::size_t>(k)];
      const double *brow = &b_full[static_cast<std::size_t>(k) * static_cast<std::size_t>(m)];
      for (int j = 0; j < m; ++j) {
        crow[static_cast<std::size_t>(j)] += aik * brow[static_cast<std::size_t>(j)];
      }
    }
  }
}

}  // namespace

bool SannikovIShtrassenAlgorithmMPI::RunImpl() {
  const auto &input = GetInput();
  const auto &a_in = std::get<0>(input);
  const auto &b_in = std::get<1>(input);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::uint64_t n0_64 = 0;
  if (rank == 0) {
    n0_64 = static_cast<std::uint64_t>(a_in.size());
  }
  MPI_Bcast(&n0_64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (!CheckSizeIsOk(n0_64)) {
    return false;
  }
  const int n0 = static_cast<int>(n0_64);
  const int m = static_cast<int>(NextPow2(static_cast<std::size_t>(n0)));
  std::vector<double> b_full;
  b_full.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  if (rank == 0) {
    PackPaddedFlatOnRoot(b_in, n0, m, &b_full);
  }
  MPI_Bcast(b_full.data(), static_cast<int>(b_full.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> a_full;
  if (rank == 0) {
    PackPaddedFlatOnRoot(a_in, n0, m, &a_full);
  }

  std::vector<int> counts_a;
  std::vector<int> displs_a;
  BuildRowPartition(size, m, m, &counts_a, &displs_a);

  const int a_local_count = counts_a[rank];
  const int local_rows = a_local_count / m;

  std::vector<double> a_local(static_cast<std::size_t>(a_local_count), 0.0);
  MPI_Scatterv(rank == 0 ? a_full.data() : nullptr, counts_a.data(), displs_a.data(), MPI_DOUBLE, a_local.data(),
               a_local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> c_local;
  MultiplyRowsClassic(a_local, local_rows, b_full, m, &c_local);

  std::vector<int> counts_c = counts_a;
  std::vector<int> displs_c = displs_a;

  std::vector<double> c_full(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  MPI_Allgatherv(c_local.data(), static_cast<int>(c_local.size()), MPI_DOUBLE, c_full.data(), counts_c.data(),
                 displs_c.data(), MPI_DOUBLE, MPI_COMM_WORLD);

  GetOutput() = UnpackCropFromFlat(c_full, m, n0);
  return !GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_shtrassen_algorithm
