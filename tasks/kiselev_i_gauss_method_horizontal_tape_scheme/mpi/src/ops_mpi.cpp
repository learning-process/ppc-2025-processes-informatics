#include "kiselev_i_gauss_method_horizontal_tape_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
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
  const auto &a_v = std::get<0>(GetInput());
  const auto &b_v = std::get<1>(GetInput());
  return !a_v.empty() && a_v.size() == b_v.size() && GetOutput().empty();
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

void Make_partition(int proc, int num, std::vector<int> &cnt, std::vector<int> &disp) {
  cnt.assign(proc, 0);
  disp.assign(proc, 0);

  int q_coef = num / proc;
  int r_coef = num % proc;
  int pos = 0;

  for (int index = 0; index < proc; ++index) {
    cnt[index] = q_coef + (index < r_coef ? 1 : 0);
    disp[index] = pos;
    pos += cnt[index];
  }
}

int Owner_of(int row, const std::vector<int> &cnt, const std::vector<int> &disp) {
  for (int proc = 0; proc < static_cast<int>(cnt.size()); ++proc) {
    if (row >= disp[proc] && row < disp[proc] + cnt[proc]) {
      return proc;
    }
  }
  return 0;
}

bool Eliminate_forward(int num, int band, int w_coef, int rank, int row0, int local_rows, const std::vector<int> &cnt,
                       const std::vector<int> &disp, std::vector<double> &a_v, std::vector<double> &b_v) {
  std::vector<double> pivot;

  for (int k_index = 0; k_index < num; ++k_index) {
    int owner = Owner_of(k_index, cnt, disp);
    int right = std::min(num - 1, k_index + band);
    int len = right - k_index + 1;

    pivot.assign(len, 0.0);
    double rhs_pivot = 0.0;

    if (rank == owner) {
      int lk = k_index - row0;
      double *row = &a_v[static_cast<std::size_t>(lk) * w_coef];
      double diag = row[band];
      if (diag == 0.0) {
        return false;
      }

      for (int j_index = k_index; j_index <= right; ++j_index) {
        pivot[j_index - k_index] = row[j_index - (k_index - band)];
      }
      rhs_pivot = b_v[static_cast<std::size_t>(lk)];
    }

    MPI_Bcast(pivot.data(), len, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    MPI_Bcast(&rhs_pivot, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    double diag = pivot[0];
    if (diag == 0.0) {
      return false;
    }

    for (int index = std::max(row0, k_index + 1); index <= std::min(row0 + local_rows - 1, k_index + band); ++index) {
      int li = index - row0;
      double *row = &a_v[static_cast<std::size_t>(li) * w_coef];

      int col = k_index - (index - band);
      if (col < 0 || col >= w_coef) {
        continue;
      }

      double m_coef = row[col] / diag;
      row[col] = 0.0;

      for (int j_index = k_index + 1; j_index <= right; ++j_index) {
        int idx = j_index - (index - band);
        if (idx >= 0 && idx < w_coef) {
          row[idx] -= m_coef * pivot[j_index - k_index];
        }
      }

      b_v[static_cast<std::size_t>(li)] -= m_coef * rhs_pivot;
    }
  }
  return true;
}

bool Eliminate_backward(int num, int band, int w_coef, int rank, int row0, const std::vector<int> &cnt,
                        const std::vector<int> &disp, const std::vector<double> &a_v, const std::vector<double> &b_v,
                        std::vector<double> &x_v) {
  x_v.assign(num, 0.0);

  for (int k_index = num - 1; k_index >= 0; --k_index) {
    int owner = Owner_of(k_index, cnt, disp);
    int right = std::min(num - 1, k_index + band);

    double val = 0.0;

    if (rank == owner) {
      int lk = k_index - row0;
      const double *row = &a_v[static_cast<std::size_t>(lk) * w_coef];

      double sum = 0.0;
      for (int j_index = k_index + 1; j_index <= right; ++j_index) {
        int idx = j_index - (k_index - band);
        if (idx >= 0 && idx < w_coef) {
          sum += row[idx] * x_v[j_index];
        }
      }

      double diag = row[band];
      if (diag == 0.0) {
        return false;
      }

      val = (b_v[static_cast<std::size_t>(lk)] - sum) / diag;
    }

    MPI_Bcast(&val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    x_v[k_index] = val;
  }
  return true;
}

}  // namespace

bool KiselevITestTaskMPI::RunImpl() {
  const auto &[a_in, b_in, band_in] = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int num = 0, band = 0;
  if (rank == 0) {
    num = static_cast<int>(a_in.size());
    band = static_cast<int>(band_in);
  }

  MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&band, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int w_coef = (2 * band) + 1;

  std::vector<int> cnt;
  std::vector<int> disp;
  Make_partition(size, num, cnt, disp);

  int local_rows = cnt[rank];
  int row0 = disp[rank];

  std::vector<double> a_loc(static_cast < std::size_t(local_rows) * static_cast < std::size_t(w_coef), 0.0);
  std::vector<double> b_loc(static_cast<std::size_t>(local_rows), 0.0);

  std::vector<int> cntA(size), dispA(size);
  for (int index = 0; index < size; ++index) {
    cntA[index] = cnt[index] * w_coef;
    dispA[index] = disp[index] * w_coef;
  }

  std::vector<double> sendA, sendB;
  if (rank == 0) {
    sendA.assign(static_cast<std::size_t>(num) * static_cast < std::size_t(w_coef), 0.0);
    sendB = b_in;

    for (int index = 0; index < num; ++index) {
      for (int j_index = std::max(0, index - band); j_index <= std::min(num - 1, index + band); ++j_index) {
        /*sendA[static_cast<std::size_t>(index * w_coef + (j_index - (index - band)))] =
         * a_in[static_cast<std::size_t>(index)][static_cast<std::size_t>(j_index)];*/
        const std::size_t rowIndex = static_cast<std::size_t>(index);
        const std::size_t colIndex = static_cast<std::size_t>(j_index);
        const std::size_t bandOffset = static_cast<std::size_t>(j_index - (index - band));
        const std::size_t linearIndex = rowIndex * static_cast<std::size_t>(w_coef) + bandOffset;

        sendA[linearIndex] = a_in[rowIndex][colIndex];
      }
    }
  }

  MPI_Scatterv(rank == 0 ? sendA.data() : nullptr, cntA.data(), dispA.data(), MPI_DOUBLE, a_loc.data(), cntA[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? sendB.data() : nullptr, cnt.data(), disp.data(), MPI_DOUBLE, b_loc.data(), local_rows,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!Eliminate_forward(num, band, w_coef, rank, row0, local_rows, cnt, disp, a_loc, b_loc)) {
    return false;
  }

  std::vector<double> x_v;
  if (!Eliminate_backward(num, band, w_coef, rank, row0, cnt, disp, a_loc, b_loc, x_v)) {
    return false;
  }

  GetOutput().swap(x_v);
  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace kiselev_i_gauss_method_horizontal_tape_scheme
