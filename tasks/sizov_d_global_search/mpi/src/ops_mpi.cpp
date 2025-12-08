#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"

namespace sizov_d_global_search {

SizovDGlobalSearchMPI::SizovDGlobalSearchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SizovDGlobalSearchMPI::ValidationImpl() {
  const auto &p = GetInput();
  if (!p.func) {
    return false;
  }
  if (p.left >= p.right) {
    return false;
  }
  if (p.accuracy <= 0.0) {
    return false;
  }
  if (p.reliability <= 0.0) {
    return false;
  }
  if (p.max_iterations <= 0) {
    return false;
  }
  return true;
}

bool SizovDGlobalSearchMPI::PreProcessingImpl() {
  const auto &p = GetInput();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int ok = 1;

  if (rank == 0) {
    x_.clear();
    y_.clear();
    x_.reserve(p.max_iterations + 10);
    y_.reserve(p.max_iterations + 10);

    double left = p.left;
    double right = p.right;
    double f_left = p.func(left);
    double f_right = p.func(right);

    if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
      ok = 0;
    } else {
      x_.push_back(left);
      x_.push_back(right);
      y_.push_back(f_left);
      y_.push_back(f_right);
      if (f_left <= f_right) {
        best_x_ = left;
        best_y_ = f_left;
      } else {
        best_x_ = right;
        best_y_ = f_right;
      }
      iterations_ = 0;
      converged_ = false;
    }
  }

  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!ok) {
    return false;
  }

  BroadcastState(0);
  return true;
}

void SizovDGlobalSearchMPI::GetChunk(std::size_t intervals, int rank, int size, std::size_t &begin, std::size_t &end) {
  if (intervals == 0U || size <= 0) {
    begin = 0;
    end = 0;
    return;
  }

  const auto s = static_cast<std::size_t>(size);
  const auto r = static_cast<std::size_t>(rank);

  const std::size_t base = intervals / s;
  const std::size_t rem = intervals % s;

  begin = r * base + std::min(r, rem);

  const auto rr = static_cast<std::size_t>(rank);
  end = begin + base + (rr < rem ? 1U : 0U);

  if (end > intervals) {
    end = intervals;
  }
}

double SizovDGlobalSearchMPI::EstimateM(double r, int rank, int size) const {
  std::size_t n = x_.size();
  if (n < 2U) {
    return r;
  }

  std::size_t intervals = n - 1U;
  std::size_t begin = 0, end = 0;
  GetChunk(intervals, rank, size, begin, end);

  double local_max = 0.0;
  for (std::size_t k = begin; k < end; ++k) {
    std::size_t i = k + 1U;
    double dx = std::abs(x_[i] - x_[i - 1U]);
    if (dx <= 0.0) {
      continue;
    }

    double y1 = y_[i];
    double y2 = y_[i - 1U];
    if (!std::isfinite(y1) || !std::isfinite(y2)) {
      continue;
    }

    double dy = std::abs(y1 - y2) / dx;
    if (std::isfinite(dy)) {
      local_max = std::max(local_max, dy);
    }
  }

  double global_max = 0.0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  constexpr double kMinSlope = 1e-2;
  return r * std::max(global_max, kMinSlope);
}

double SizovDGlobalSearchMPI::Characteristic(std::size_t i, double m) const {
  double xr = x_[i], xl = x_[i - 1U];
  double yr = y_[i], yl = y_[i - 1U];
  double dx = xr - xl;
  double df = yr - yl;
  return (m * dx) + (df * df) / (m * dx) - 2.0 * (yr + yl);
}

double SizovDGlobalSearchMPI::NewPoint(std::size_t i, double m) const {
  double xr = x_[i], xl = x_[i - 1U];
  double yr = y_[i], yl = y_[i - 1U];
  double mid = 0.5 * (xr + xl);
  double shift = (yr - yl) / (2.0 * m);
  double x_new = mid - shift;
  if (x_new <= xl || x_new >= xr) {
    return mid;
  }
  return x_new;
}

SizovDGlobalSearchMPI::IntervalChar SizovDGlobalSearchMPI::ComputeLocalBestInterval(double m, int rank,
                                                                                    int size) const {
  IntervalChar r{};
  r.characteristic = -std::numeric_limits<double>::infinity();
  r.index = -1;

  std::size_t n = x_.size();
  if (n < 2U) {
    return r;
  }

  std::size_t intervals = n - 1U;
  std::size_t begin = 0, end = 0;
  GetChunk(intervals, rank, size, begin, end);

  for (std::size_t k = begin; k < end; ++k) {
    std::size_t i = k + 1U;
    double c = Characteristic(i, m);
    if (c > r.characteristic) {
      r.characteristic = c;
      r.index = static_cast<int>(i);
    }
  }
  return r;
}

int SizovDGlobalSearchMPI::ReduceBestIntervalIndex(const IntervalChar &local, int n) {
  IntervalChar global{};
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  if (global.index < 1 || global.index >= n) {
    return -1;
  }
  return global.index;
}

bool SizovDGlobalSearchMPI::CheckStopByAccuracy(const Problem &p, int best_idx, int rank) {
  bool stop = false;
  if (rank == 0) {
    double L = x_[best_idx - 1U];
    double R = x_[best_idx];
    if ((R - L) <= p.accuracy) {
      converged_ = true;
      stop = true;
    }
  }
  int f = stop ? 1 : 0;
  MPI_Bcast(&f, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return f;
}

void SizovDGlobalSearchMPI::InsertNewPoint(const Problem &p, std::size_t best_idx, double m, int rank) {
  if (rank != 0) {
    return;
  }

  double x_new = NewPoint(best_idx, m);
  double y_new = p.func(x_new);
  if (!std::isfinite(y_new)) {
    return;
  }

  auto pos = std::ranges::lower_bound(x_, x_new);
  std::size_t idx = pos - x_.begin();
  x_.insert(pos, x_new);
  y_.insert(y_.begin() + idx, y_new);

  if (y_new < best_y_) {
    best_y_ = y_new;
    best_x_ = x_new;
  }
}

struct InsertMsg {
  double x_new;
  double y_new;
  int idx;
};

void SizovDGlobalSearchMPI::BroadcastState(int rank) {
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(x_.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  x_.resize(n);
  y_.resize(n);

  MPI_Bcast(x_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(y_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void SizovDGlobalSearchMPI::BroadcastResult(int rank) {
  std::array<double, 2> r{};
  std::array<int, 2> m{};
  if (rank == 0) {
    r[0] = best_x_;
    r[1] = best_y_;
    m[0] = iterations_;
    m[1] = converged_ ? 1 : 0;
  }
  MPI_Bcast(r.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(m.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  best_x_ = r[0];
  best_y_ = r[1];
  iterations_ = m[0];
  converged_ = (m[1] != 0);
}

bool SizovDGlobalSearchMPI::RunImpl() {
  const auto &p = GetInput();

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (x_.size() < 2U) {
    return false;
  }

  double m = EstimateM(p.reliability, rank, size);

  for (int iter = 0; iter < p.max_iterations; ++iter) {
    iterations_ = iter + 1;

    if (iter % 5 == 0) {
      m = EstimateM(p.reliability, rank, size);
    }

    int n = static_cast<int>(x_.size());
    IntervalChar local = ComputeLocalBestInterval(m, rank, size);
    int best_idx = ReduceBestIntervalIndex(local, n);
    if (best_idx < 1) {
      converged_ = false;
      break;
    }

    if (CheckStopByAccuracy(p, best_idx, rank)) {
      break;
    }

    InsertMsg msg{};
    if (rank == 0) {
      double x_new = NewPoint(best_idx, m);
      double y_new = p.func(x_new);
      auto pos = std::ranges::lower_bound(x_, x_new);
      int idx = static_cast<int>(pos - x_.begin());
      msg.x_new = x_new;
      msg.y_new = y_new;
      msg.idx = idx;

      if (std::isfinite(y_new)) {
        x_.insert(pos, x_new);
        y_.insert(y_.begin() + idx, y_new);
        if (y_new < best_y_) {
          best_x_ = x_new;
          best_y_ = y_new;
        }
      } else {
        msg.idx = -1;
      }
    }

    MPI_Bcast(&msg, sizeof(InsertMsg), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank != 0 && msg.idx >= 0) {
      auto pos = x_.begin() + msg.idx;
      x_.insert(pos, msg.x_new);
      y_.insert(y_.begin() + msg.idx, msg.y_new);
    }
  }

  BroadcastResult(rank);
  GetOutput() = Solution{best_x_, best_y_, iterations_, converged_};
  return true;
}

bool SizovDGlobalSearchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_global_search
