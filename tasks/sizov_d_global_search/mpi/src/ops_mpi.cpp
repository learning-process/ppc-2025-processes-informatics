#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>  // DEBUG
#include <limits>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"

namespace {

// ───────────────────────────── ВСПОМОГАТЕЛЬНЫЕ СТРУКТУРЫ ──────────────────────

struct IntervalCharacteristic {
  double characteristic = -std::numeric_limits<double>::infinity();
  int right_index = 1;
};

double EstimateM(const std::vector<double> &points, const std::vector<double> &values, double reliability) {
  assert(points.size() == values.size());
  if (points.size() < 2) {
    return 1.0;
  }

  double max_slope = 0.0;
  for (std::size_t i = 1; i < points.size(); ++i) {
    const double dx = points[i] - points[i - 1];
    if (dx <= 0.0) {
      continue;
    }

    const double dv = values[i] - values[i - 1];
    const double slope = std::abs(dv / dx);
    if (std::isfinite(slope)) {
      max_slope = std::max(max_slope, slope);
    }
  }

  if (max_slope <= 0.0 || !std::isfinite(max_slope)) {
    return 1.0;
  }

  const double m = reliability * max_slope;
  if (!std::isfinite(m) || m <= 0.0) {
    return 1.0;
  }

  return m;
}

double CalcCharacteristic(const std::vector<double> &points, const std::vector<double> &values, std::size_t right_idx,
                          double m) {
  assert(right_idx > 0);
  assert(right_idx < points.size());
  assert(points.size() == values.size());

  const double dx = points[right_idx] - points[right_idx - 1];
  if (dx <= std::numeric_limits<double>::epsilon()) {
    return -std::numeric_limits<double>::infinity();
  }

  const double df = values[right_idx] - values[right_idx - 1];
  const double term1 = m * dx;
  const double denom = m * dx;

  if (denom <= 0.0 || !std::isfinite(denom)) {
    return -std::numeric_limits<double>::infinity();
  }

  const double term2 = (df * df) / denom;
  const double penalty = 2.0 * (values[right_idx] + values[right_idx - 1]);
  const double result = term1 + term2 - penalty;

  if (!std::isfinite(result)) {
    return -std::numeric_limits<double>::infinity();
  }
  return result;
}

double CalcNewPoint(const std::vector<double> &points, const std::vector<double> &values, std::size_t right_idx,
                    double m) {
  assert(right_idx > 0);
  assert(right_idx < points.size());
  assert(points.size() == values.size());

  const double df = values[right_idx] - values[right_idx - 1];
  const double mid = 0.5 * (points[right_idx] + points[right_idx - 1]);
  const double shift = df / (2.0 * m);
  const double candidate = mid - shift;

  if (!std::isfinite(candidate)) {
    return mid;
  }
  return candidate;
}

std::pair<std::size_t, std::size_t> GetChunk(std::size_t total_work, int size, int rank) {
  if (size <= 0 || total_work == 0) {
    return {0, 0};
  }

  const std::size_t s = static_cast<std::size_t>(size);
  const std::size_t r = static_cast<std::size_t>(rank);

  const std::size_t base = total_work / s;
  const std::size_t remainder = total_work % s;

  const std::size_t begin = r * base + std::min<std::size_t>(remainder, r);
  const std::size_t end = begin + base + (r < remainder ? 1 : 0);

  return {begin, std::min(end, total_work)};
}

}  // namespace

namespace sizov_d_global_search {

SizovDGlobalSearchMPI::SizovDGlobalSearchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SizovDGlobalSearchMPI::ValidationImpl() {
  const auto &problem = GetInput();

  if (!problem.func) {
    return false;
  }
  if (!(problem.left < problem.right)) {
    return false;
  }
  if (!(problem.accuracy > 0.0) || !std::isfinite(problem.accuracy)) {
    return false;
  }
  if (!(problem.reliability > 0.0) || !std::isfinite(problem.reliability)) {
    return false;
  }
  if (problem.max_iterations <= 0) {
    return false;
  }

  return true;
}

bool SizovDGlobalSearchMPI::PreProcessingImpl() {
  const auto &problem = GetInput();

  points_.clear();
  values_.clear();
  iterations_ = 0;
  converged_ = false;

  const double left_value = problem.func(problem.left);
  const double right_value = problem.func(problem.right);

  points_.push_back(problem.left);
  points_.push_back(problem.right);
  values_.push_back(left_value);
  values_.push_back(right_value);

  if (left_value <= right_value) {
    best_point_ = problem.left;
    best_value_ = left_value;
  } else {
    best_point_ = problem.right;
    best_value_ = right_value;
  }

  GetOutput() = {best_point_, best_value_, 0, false};
  return true;
}

bool SizovDGlobalSearchMPI::RunImpl() {
  const auto &problem = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "[DEBUG][ops_mpi.cpp] START: left=" << problem.left << " right=" << problem.right
              << " accuracy=" << problem.accuracy << " reliability=" << problem.reliability
              << " max_iter=" << problem.max_iterations << "\n";
  }

  if (points_.size() < 2) {
    converged_ = false;
    GetOutput() = {best_point_, best_value_, 0, false};
    return true;
  }

  int performed_iterations = 0;

  for (int iter = 0; iter < problem.max_iterations; ++iter) {
    performed_iterations = iter + 1;

    const double m = EstimateM(points_, values_, problem.reliability);

    if (rank == 0) {
      std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] M = " << m << " points=" << points_.size() << "\n";
    }

    if (!(m > 0.0) || !std::isfinite(m)) {
      if (rank == 0) {
        std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] ERROR: invalid M value\n";
      }
      converged_ = false;
      break;
    }

    const std::size_t interval_count = points_.size() - 1;

    if (interval_count == 0) {
      converged_ = false;
      break;
    }

    const auto [chunk_begin, chunk_end] = GetChunk(interval_count, size, rank);

    IntervalCharacteristic local;

    if (chunk_begin < chunk_end) {
      for (std::size_t interval = chunk_begin; interval < chunk_end; ++interval) {
        const std::size_t right_idx = interval + 1;
        const double characteristic = CalcCharacteristic(points_, values_, right_idx, m);

        if (characteristic > local.characteristic) {
          local.characteristic = characteristic;
          local.right_index = static_cast<int>(right_idx);
        }
      }
    }

    IntervalCharacteristic global{};
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    std::size_t best_right_idx = 1;
    if (global.right_index > 0 && static_cast<std::size_t>(global.right_index) < points_.size()) {
      best_right_idx = static_cast<std::size_t>(global.right_index);
    }

    if (rank == 0) {
      std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] Best interval = (" << points_[best_right_idx - 1] << ", "
                << points_[best_right_idx] << ") idx=" << best_right_idx << " characteristic=" << global.characteristic
                << "\n";
    }

    const double interval_length = points_[best_right_idx] - points_[best_right_idx - 1];

    if (interval_length <= problem.accuracy) {
      if (rank == 0) {
        std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] STOP: interval length <= accuracy\n";
      }
      converged_ = true;
      break;
    }

    const double new_point = CalcNewPoint(points_, values_, best_right_idx, m);
    const double new_value = problem.func(new_point);

    if (rank == 0) {
      std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] New point x=" << new_point << " f(x)=" << new_value
                << "\n";
    }

    if (new_point < problem.left || new_point > problem.right || !std::isfinite(new_point) ||
        !std::isfinite(new_value)) {
      if (rank == 0) {
        std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] ERROR: new point out of bounds or NaN\n";
      }
      converged_ = false;
      break;
    }

    const auto insert_pos = static_cast<std::ptrdiff_t>(best_right_idx);
    points_.insert(points_.begin() + insert_pos, new_point);
    values_.insert(values_.begin() + insert_pos, new_value);

    if (new_value < best_value_) {
      if (rank == 0) {
        std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] New BEST: x=" << new_point << " f=" << new_value
                  << "\n";
      }
      best_point_ = new_point;
      best_value_ = new_value;
    }
  }

  iterations_ = performed_iterations;

  if (rank == 0) {
    std::cout << "[DEBUG][ops_mpi.cpp] FINISH: converged=" << converged_ << " best_x=" << best_point_
              << " best_f=" << best_value_ << " iterations=" << iterations_ << "\n";
  }

  GetOutput() = {best_point_, best_value_, iterations_, converged_};
  return true;
}

bool SizovDGlobalSearchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_global_search
