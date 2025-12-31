#include "kiselev_i_gauss_method_horizontal_tape_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "tasks/kiselev_i_gauss_method_horizontal_tape_scheme/common/include/common.hpp"

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &buf = GetInput();
  InType tmp(in);
  buf.swap(tmp);
  GetOutput().clear();
}

bool KiselevITestTaskMPI::ValidationImpl() {
  const auto &aVector = std::get<0>(GetInput());
  const auto &bVector = std::get<1>(GetInput());
  return !aVector.empty() && aVector.size() == bVector.size() && GetOutput().empty();
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

void MakePartition(int proc, int num, std::vector<int> &cnt, std::vector<int> &disp) {
  cnt.assign(proc, 0);
  disp.assign(proc, 0);

  int qCoef = num / proc;
  int rCoef = num % proc;
  int pos = 0;

  for (int index = 0; index < proc; ++index) {
    cnt[index] = qCoef + (index < rCoef ? 1 : 0);
    disp[index] = pos;
    pos += cnt[index];
  }
}

int OwnerOf(int row, const std::vector<int> &cnt, const std::vector<int> &disp) {
  for (int proc = 0; proc < static_cast<int>(cnt.size()); ++proc) {
    if (row >= disp[proc] && row < disp[proc] + cnt[proc]) {
      return proc;
    }
  }
  return 0;
}

bool BuildPivotRow(int k_index, int band, int w_coef, int rank, int owner, int row0,
                   const std::vector<double> &a_vector, const std::vector<double> &b_vector, std::vector<double> &pivot,
                   double &rhs_pivot) {
  if (rank != owner) {
    return true;
  }

  const int local_k = k_index - row0;
  const auto row_offset = static_cast<std::size_t>(local_k) * static_cast<std::size_t>(w_coef);
  const double *row = &a_vector[row_offset];

  const double diag = row[band];
  if (diag == 0.0) {
    return false;
  }

  const int right = static_cast<int>(pivot.size()) - 1 + k_index;
  for (int j = k_index; j <= right; ++j) {
    pivot[j - k_index] = row[j - (k_index - band)];
  }

  rhs_pivot = b_vector[static_cast<std::size_t>(local_k)];
  return true;
}

bool ApplyElimination(int k_index, int band, int w_coef, int row0, int local_rows, const std::vector<double> &pivot,
                      double rhs_pivot, std::vector<double> &a_vector, std::vector<double> &b_vector) {
  const double diag = pivot[0];
  if (diag == 0.0) {
    return false;
  }

  const int right = k_index + static_cast<int>(pivot.size()) - 1;
  const int row_begin = std::max(row0, k_index + 1);
  const int row_end = std::min(row0 + local_rows - 1, k_index + band);

  for (int index = row_begin; index <= row_end; ++index) {
    const int local_i = index - row0;
    const auto row_offset = static_cast<std::size_t>(local_i) * static_cast<std::size_t>(w_coef);
    double *row = &a_vector[row_offset];

    const int col = k_index - (index - band);
    if (col < 0 || col >= w_coef) {
      continue;
    }

    const double multiplier = row[col] / diag;
    row[col] = 0.0;

    for (int j = k_index + 1; j <= right; ++j) {
      const int idx = j - (index - band);
      if (idx >= 0 && idx < w_coef) {
        row[idx] -= multiplier * pivot[j - k_index];
      }
    }

    b_vector[static_cast<std::size_t>(local_i)] -= multiplier * rhs_pivot;
  }

  return true;
}

bool EliminateForward(int num, int band, int wCoef, int rank, int row0, int localRows, const std::vector<int> &cnt,
                      const std::vector<int> &disp, std::vector<double> &aVector, std::vector<double> &bVector) {
  for (int kIndex = 0; kIndex < num; ++kIndex) {
    const int owner = OwnerOf(kIndex, cnt, disp);
    const int right = std::min(num - 1, kIndex + band);
    const int length = right - kIndex + 1;

    std::vector<double> pivot(static_cast<std::size_t>(length), 0.0);
    double rhsPivot = 0.0;

    if (!BuildPivotRow(kIndex, band, wCoef, rank, owner, row0, aVector, bVector, pivot, rhsPivot)) {
      return false;
    }

    MPI_Bcast(pivot.data(), length, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    MPI_Bcast(&rhsPivot, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    if (!ApplyElimination(kIndex, band, wCoef, row0, localRows, pivot, rhsPivot, aVector, bVector)) {
      return false;
    }
  }

  return true;
}

bool EliminateBackward(int num, int band, int wCoef, int rank, int row0, const std::vector<int> &cnt,
                       const std::vector<int> &disp, const std::vector<double> &aVector,
                       const std::vector<double> &bVector, std::vector<double> &xVector) {
  xVector.assign(num, 0.0);

  for (int kIndex = num - 1; kIndex >= 0; --kIndex) {
    int owner = OwnerOf(kIndex, cnt, disp);
    int right = std::min(num - 1, kIndex + band);

    double val = 0.0;

    if (rank == owner) {
      int lk = kIndex - row0;
      const double *row = &aVector[static_cast<std::size_t>(lk) * wCoef];

      double sum = 0.0;
      for (int jIndex = kIndex + 1; jIndex <= right; ++jIndex) {
        int idx = jIndex - (kIndex - band);
        if (idx >= 0 && idx < wCoef) {
          sum += row[idx] * xVector[jIndex];
        }
      }

      double diag = row[band];
      if (diag == 0.0) {
        return false;
      }

      val = (bVector[static_cast<std::size_t>(lk)] - sum) / diag;
    }

    MPI_Bcast(&val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    xVector[kIndex] = val;
  }
  return true;
}

}  // namespace

bool KiselevITestTaskMPI::RunImpl() {
  const auto &[aIn, bIn, band_in] = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int num = 0;
  int band = 0;
  if (rank == 0) {
    num = static_cast<int>(aIn.size());
    band = static_cast<int>(band_in);
  }

  MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&band, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int wCoef = (2 * band) + 1;

  std::vector<int> cnt;
  std::vector<int> disp;
  MakePartition(size, num, cnt, disp);

  int localRows = cnt[rank];
  int row0 = disp[rank];

  std::vector<double> aLoc(static_cast<std::size_t>(localRows) * static_cast<std::size_t>(wCoef), 0.0);
  std::vector<double> bLoc(static_cast<std::size_t>(localRows), 0.0);

  std::vector<int> cntA(size);
  std::vector<int> dispA(size);
  for (int index = 0; index < size; ++index) {
    cntA[index] = cnt[index] * wCoef;
    dispA[index] = disp[index] * wCoef;
  }

  std::vector<double> sendA;
  std::vector<double> sendB;
  if (rank == 0) {
    sendA.assign(static_cast<std::size_t>(num) * static_cast<std::size_t>(wCoef), 0.0);
    sendB = bIn;

    for (int index = 0; index < num; ++index) {
      for (int jIndex = std::max(0, index - band); jIndex <= std::min(num - 1, index + band); ++jIndex) {
        // tmp
        const auto rowIndex = static_cast<std::size_t>(index);
        const std::size_t colIndex = static_cast<std::size_t>(jIndex);
        const std::size_t bandOffset = static_cast<std::size_t>(jIndex - (index - band));
        const std::size_t linearIndex = (rowIndex * static_cast<std::size_t>(wCoef)) + bandOffset;

        sendA[linearIndex] = aIn[rowIndex][colIndex];
      }
    }
  }

  MPI_Scatterv(rank == 0 ? sendA.data() : nullptr, cntA.data(), dispA.data(), MPI_DOUBLE, aLoc.data(), cntA[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? sendB.data() : nullptr, cnt.data(), disp.data(), MPI_DOUBLE, bLoc.data(), localRows,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!EliminateForward(num, band, wCoef, rank, row0, localRows, cnt, disp, aLoc, bLoc)) {
    return false;
  }

  std::vector<double> xVector;
  if (!EliminateBackward(num, band, wCoef, rank, row0, cnt, disp, aLoc, bLoc, xVector)) {
    return false;
  }

  GetOutput().swap(xVector);
  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
