#include "kiselev_i_gauss_method_horizontal_tape_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

namespace kiselev_i_gauss_method_horizontal_tape_scheme {

KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &buf = GetInput();
  InType tmp(in);
  buf.swap(tmp);
  GetOutput().clear();
}

bool KiselevITestTaskMPI::ValidationImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  return !a.empty() && a.size() == b.size() && GetOutput().empty();
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

void make_partition(int p, int n, std::vector<int> &cnt, std::vector<int> &disp) {
  cnt.assign(p, 0);
  disp.assign(p, 0);

  int q = n / p;
  int r = n % p;
  int pos = 0;

  for (int i = 0; i < p; ++i) {
    cnt[i] = q + (i < r ? 1 : 0);
    disp[i] = pos;
    pos += cnt[i];
  }
}

int owner_of(int row, const std::vector<int> &cnt, const std::vector<int> &disp) {
  for (int p = 0; p < static_cast<int>(cnt.size()); ++p) {
    if (row >= disp[p] && row < disp[p] + cnt[p]) {
      return p;
    }
  }
  return 0;
}

bool eliminate_forward(int n, int band, int w, int rank, int row0, int local_rows, const std::vector<int> &cnt,
                       const std::vector<int> &disp, std::vector<double> &a, std::vector<double> &b) {
  std::vector<double> pivot;

  for (int k = 0; k < n; ++k) {
    int owner = owner_of(k, cnt, disp);
    int right = std::min(n - 1, k + band);
    int len = right - k + 1;

    pivot.assign(len, 0.0);
    double rhs_pivot = 0.0;

    if (rank == owner) {
      int lk = k - row0;
      double *row = &a[static_cast<std::size_t>(lk) * w];
      double diag = row[band];
      if (diag == 0.0) {
        return false;
      }

      for (int j = k; j <= right; ++j) {
        pivot[j - k] = row[j - (k - band)];
      }
      rhs_pivot = b[static_cast<std::size_t>(lk)];
    }

    MPI_Bcast(pivot.data(), len, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    MPI_Bcast(&rhs_pivot, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    double diag = pivot[0];
    if (diag == 0.0) {
      return false;
    }

    for (int i = std::max(row0, k + 1); i <= std::min(row0 + local_rows - 1, k + band); ++i) {
      int li = i - row0;
      double *row = &a[static_cast<std::size_t>(li) * w];

      int col = k - (i - band);
      if (col < 0 || col >= w) {
        continue;
      }

      double m = row[col] / diag;
      row[col] = 0.0;

      for (int j = k + 1; j <= right; ++j) {
        int idx = j - (i - band);
        if (idx >= 0 && idx < w) {
          row[idx] -= m * pivot[j - k];
        }
      }

      b[static_cast<std::size_t>(li)] -= m * rhs_pivot;
    }
  }
  return true;
}

bool eliminate_backward(int n, int band, int w, int rank, int row0, const std::vector<int> &cnt,
                        const std::vector<int> &disp, const std::vector<double> &a, const std::vector<double> &b,
                        std::vector<double> &x) {
  x.assign(n, 0.0);

  for (int k = n - 1; k >= 0; --k) {
    int owner = owner_of(k, cnt, disp);
    int right = std::min(n - 1, k + band);

    double val = 0.0;

    if (rank == owner) {
      int lk = k - row0;
      const double *row = &a[static_cast<std::size_t>(lk) * w];

      double sum = 0.0;
      for (int j = k + 1; j <= right; ++j) {
        int idx = j - (k - band);
        if (idx >= 0 && idx < w) {
          sum += row[idx] * x[j];
        }
      }

      double diag = row[band];
      if (diag == 0.0) {
        return false;
      }

      val = (b[static_cast<std::size_t>(lk)] - sum) / diag;
    }

    MPI_Bcast(&val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    x[k] = val;
  }
  return true;
}

}  // namespace

bool KiselevITestTaskMPI::RunImpl() {
  const auto &[a_in, b_in, band_in] = GetInput();

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0, band = 0;
  if (rank == 0) {
    n = static_cast<int>(a_in.size());
    band = static_cast<int>(band_in);
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&band, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int w = 2 * band + 1;

  std::vector<int> cnt, disp;
  make_partition(size, n, cnt, disp);

  int local_rows = cnt[rank];
  int row0 = disp[rank];

  std::vector<double> a_loc(static_cast<std::size_t>(local_rows * w), 0.0);
  std::vector<double> b_loc(static_cast<std::size_t>(local_rows), 0.0);

  std::vector<int> cntA(size), dispA(size);
  for (int i = 0; i < size; ++i) {
    cntA[i] = cnt[i] * w;
    dispA[i] = disp[i] * w;
  }

  std::vector<double> sendA, sendB;
  if (rank == 0) {
    sendA.assign(static_cast<std::size_t>(n * w), 0.0);
    sendB = b_in;

    for (int i = 0; i < n; ++i) {
      for (int j = std::max(0, i - band); j <= std::min(n - 1, i + band); ++j) {
        sendA[static_cast<std::size_t>(i * w + (j - (i - band)))] =
            a_in[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      }
    }
  }

  MPI_Scatterv(rank == 0 ? sendA.data() : nullptr, cntA.data(), dispA.data(), MPI_DOUBLE, a_loc.data(), cntA[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? sendB.data() : nullptr, cnt.data(), disp.data(), MPI_DOUBLE, b_loc.data(), local_rows,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!eliminate_forward(n, band, w, rank, row0, local_rows, cnt, disp, a_loc, b_loc)) {
    return false;
  }

  std::vector<double> x;
  if (!eliminate_backward(n, band, w, rank, row0, cnt, disp, a_loc, b_loc, x)) {
    return false;
  }

  GetOutput().swap(x);
  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
