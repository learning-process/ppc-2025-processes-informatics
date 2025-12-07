#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
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
    const double left = p.left;
    const double right = p.right;

    x_.clear();
    y_.clear();

    x_.push_back(left);
    x_.push_back(right);

    const double f_left = p.func(left);
    const double f_right = p.func(right);

    if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
      ok = 0;
    } else {
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
  if (ok == 0) {
    return false;
  }

  BroadcastState(rank);
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

  begin = (r * base) + std::min(r, rem);
  end = begin + base + (r < rem ? 1U : 0U);
  end = std::min(end, intervals);
}

double SizovDGlobalSearchMPI::EstimateM(double r, int rank, int size) const {
  const auto n = x_.size();
  if (n < 2U) {
    return r;
  }

  const auto intervals = n - 1U;

  std::size_t begin = 0;
  std::size_t end = 0;
  GetChunk(intervals, rank, size, begin, end);

  double local_max = 0.0;

  for (std::size_t k = begin; k < end; ++k) {
    const std::size_t i = k + 1U;
    const double dx = std::abs(x_[i] - x_[i - 1U]);
    if (dx <= 0.0) {
      continue;
    }

    const double y_r = y_[i];
    const double y_l = y_[i - 1U];

    if (!std::isfinite(y_r) || !std::isfinite(y_l)) {
      continue;
    }

    const double dy = std::abs(y_r - y_l) / dx;
    if (std::isfinite(dy)) {
      local_max = std::max(local_max, dy);
    }
  }

  double global_max = 0.0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  constexpr double kMinimalSlope = 1e-2;
  if (global_max < kMinimalSlope) {
    global_max = kMinimalSlope;
  }

  return r * global_max;
}

double SizovDGlobalSearchMPI::Characteristic(std::size_t i, double m) const {
  const double xi = x_[i];
  const double xi_prev = x_[i - 1U];
  const double yi = y_[i];
  const double yi_prev = y_[i - 1U];

  const double dx = xi - xi_prev;
  const double df = yi - yi_prev;

  const double term1 = m * dx;
  const double term2 = (df * df) / (m * dx);
  const double term3 = 2.0 * (yi + yi_prev);

  return term1 + term2 - term3;
}

double SizovDGlobalSearchMPI::NewPoint(std::size_t i, double m) const {
  const double xi = x_[i];
  const double xi_prev = x_[i - 1U];
  const double yi = y_[i];
  const double yi_prev = y_[i - 1U];

  const double mid = 0.5 * (xi + xi_prev);
  const double shift = (yi - yi_prev) / (2.0 * m);

  double x_new = mid - shift;
  if (x_new <= xi_prev || x_new >= xi) {
    x_new = mid;
  }

  return x_new;
}

void SizovDGlobalSearchMPI::BroadcastState(int rank) {
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(x_.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  x_.resize(static_cast<std::size_t>(n));
  y_.resize(static_cast<std::size_t>(n));

  if (n > 0) {
    MPI_Bcast(x_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

SizovDGlobalSearchMPI::IntervalChar SizovDGlobalSearchMPI::ComputeLocalBestInterval(double m, int rank,
                                                                                    int size) const {
  IntervalChar local{};
  local.characteristic = -std::numeric_limits<double>::infinity();
  local.index = -1;

  const auto n = x_.size();
  if (n < 2U) {
    return local;
  }

  const auto intervals = n - 1U;

  std::size_t begin = 0;
  std::size_t end = 0;
  GetChunk(intervals, rank, size, begin, end);

  for (std::size_t k = begin; k < end; ++k) {
    const std::size_t i = k + 1U;
    const double c = Characteristic(i, m);
    if (c > local.characteristic) {
      local.characteristic = c;
      local.index = static_cast<int>(i);
    }
  }

  return local;
}

int SizovDGlobalSearchMPI::ReduceBestIntervalIndex(const IntervalChar &local, int n) {
  IntervalChar global{};
  global.characteristic = -std::numeric_limits<double>::infinity();
  global.index = -1;

  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  if (global.index < 1 || global.index >= n) {
    return -1;
  }

  return global.index;
}

bool SizovDGlobalSearchMPI::CheckStopByAccuracy(const Problem &p, int best_idx_int, int rank) {
  bool stop_now = false;

  if (rank == 0) {
    const auto idx = static_cast<std::size_t>(best_idx_int);
    const double left = x_[idx - 1U];
    const double right = x_[idx];

    if ((right - left) <= p.accuracy) {
      converged_ = true;
      stop_now = true;
    }
  }

  int stop_flag = stop_now ? 1 : 0;
  MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return stop_flag != 0;
}

void SizovDGlobalSearchMPI::InsertNewPoint(const Problem &p, std::size_t best_idx, double m, int rank) {
  if (rank != 0) {
    return;
  }

  const double x_new = NewPoint(best_idx, m);
  const double y_new = p.func(x_new);

  if (!std::isfinite(y_new)) {
    return;
  }

  auto pos = std::ranges::lower_bound(x_, x_new);
  const auto idx = static_cast<std::size_t>(std::distance(x_.begin(), pos));

  x_.insert(pos, x_new);
  y_.insert(y_.begin() + static_cast<std::ptrdiff_t>(idx), y_new);

  if (iterations_ == 1 || y_new < best_y_) {
    best_x_ = x_new;
    best_y_ = y_new;
  }
}

void SizovDGlobalSearchMPI::BroadcastResult(int rank) {
  std::array<double, 2> result{};
  std::array<int, 2> meta{};

  if (rank == 0) {
    result[0] = best_x_;
    result[1] = best_y_;
    meta[0] = iterations_;
    meta[1] = converged_ ? 1 : 0;
  }

  MPI_Bcast(result.data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(meta.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  best_x_ = result[0];
  best_y_ = result[1];
  iterations_ = meta[0];
  converged_ = (meta[1] != 0);
}

bool SizovDGlobalSearchMPI::RunImpl() {
  const auto &p = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (x_.size() < 2U) {
    return false;
  }

  for (int iter = 0; iter < p.max_iterations; ++iter) {
    iterations_ = iter + 1;

    BroadcastState(rank);

    const auto n = x_.size();
    if (n < 2U) {
      converged_ = false;
      break;
    }

    const double m = EstimateM(p.reliability, rank, size);

    const IntervalChar local = ComputeLocalBestInterval(m, rank, size);
    const int best_idx_int = ReduceBestIntervalIndex(local, static_cast<int>(n));

    if (best_idx_int < 1) {
      converged_ = false;
      break;
    }

    if (CheckStopByAccuracy(p, best_idx_int, rank)) {
      break;
    }

    InsertNewPoint(p, static_cast<std::size_t>(best_idx_int), m, rank);
  }

  BroadcastResult(rank);

  GetOutput() = Solution{
      .argmin = best_x_,
      .value = best_y_,
      .iterations = iterations_,
      .converged = converged_,
  };

  return true;
}

bool SizovDGlobalSearchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_global_search
