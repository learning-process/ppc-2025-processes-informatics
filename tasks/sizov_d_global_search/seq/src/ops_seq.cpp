#include "sizov_d_global_search/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"

namespace sizov_d_global_search {

SizovDGlobalSearchSEQ::SizovDGlobalSearchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SizovDGlobalSearchSEQ::ValidationImpl() {
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

bool SizovDGlobalSearchSEQ::PreProcessingImpl() {
  const auto &p = GetInput();

  x_.clear();
  y_.clear();

  x_.reserve(static_cast<std::size_t>(p.max_iterations) + 2U);
  y_.reserve(static_cast<std::size_t>(p.max_iterations) + 2U);

  const double f_left = p.func(p.left);
  const double f_right = p.func(p.right);

  if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
    return false;
  }

  x_.push_back(p.left);
  x_.push_back(p.right);
  y_.push_back(f_left);
  y_.push_back(f_right);

  if (f_left <= f_right) {
    best_x_ = p.left;
    best_y_ = f_left;
  } else {
    best_x_ = p.right;
    best_y_ = f_right;
  }

  iterations_ = 0;
  converged_ = false;

  return true;
}

double SizovDGlobalSearchSEQ::EstimateM(double reliability) const {
  double max_slope = 0.0;

  const std::size_t n = x_.size();
  for (std::size_t i = 1; i < n; ++i) {
    const double dx = std::abs(x_[i] - x_[i - 1U]);
    if (dx <= 0.0) {
      continue;
    }

    const double dy = std::abs(y_[i] - y_[i - 1U]) / dx;
    if (std::isfinite(dy)) {
      max_slope = std::max(max_slope, dy);
    }
  }

  constexpr double kMinimalSlope = 1e-2;
  return reliability * std::max(max_slope, kMinimalSlope);
}

double SizovDGlobalSearchSEQ::Characteristic(std::size_t idx, double m) const {
  const double xl = x_[idx - 1U];
  const double xr = x_[idx];
  const double yl = y_[idx - 1U];
  const double yr = y_[idx];

  const double dx = xr - xl;
  const double df = yr - yl;

  return (m * dx) + (df * df) / (m * dx) - 2.0 * (yr + yl);
}

double SizovDGlobalSearchSEQ::NewPoint(std::size_t idx, double m) const {
  const double xl = x_[idx - 1U];
  const double xr = x_[idx];
  const double yl = y_[idx - 1U];
  const double yr = y_[idx];

  const double mid = 0.5 * (xl + xr);
  const double shift = (yr - yl) / (2.0 * m);

  double x_new = mid - shift;
  if (x_new <= xl || x_new >= xr) {
    x_new = mid;
  }

  return x_new;
}

bool SizovDGlobalSearchSEQ::RunImpl() {
  const auto &p = GetInput();

  if (x_.size() < 2U) {
    return false;
  }

  double m = EstimateM(p.reliability);

  for (int iter = 0; iter < p.max_iterations; ++iter) {
    iterations_ = iter + 1;

    if ((iter % 5) == 0) {
      m = EstimateM(p.reliability);
    }

    const std::size_t n = x_.size();
    if (n < 2U) {
      converged_ = false;
      break;
    }

    double best_char = -std::numeric_limits<double>::infinity();
    std::size_t best_idx = 1U;

    for (std::size_t i = 1; i < n; ++i) {
      const double c = Characteristic(i, m);
      if (c > best_char) {
        best_char = c;
        best_idx = i;
      }
    }

    const double L = x_[best_idx - 1U];
    const double R = x_[best_idx];
    const double width = R - L;

    if (width <= p.accuracy) {
      converged_ = true;
      break;
    }

    const double x_new = NewPoint(best_idx, m);
    const double y_new = p.func(x_new);

    if (!std::isfinite(y_new)) {
      continue;
    }

    auto pos = std::ranges::lower_bound(x_, x_new);
    const std::size_t idx = static_cast<std::size_t>(pos - x_.begin());

    x_.insert(pos, x_new);
    y_.insert(y_.begin() + static_cast<std::ptrdiff_t>(idx), y_new);

    if (y_new < best_y_) {
      best_y_ = y_new;
      best_x_ = x_new;
    }
  }

  GetOutput() = Solution{best_x_, best_y_, iterations_, converged_};
  return true;
}

bool SizovDGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_global_search
