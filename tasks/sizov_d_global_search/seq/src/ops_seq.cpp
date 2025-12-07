#include "sizov_d_global_search/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"

#define DBG_ENABLED (std::getenv("DEBUG_GLOBAL_SEARCH") != nullptr)
#define DBG(msg)                                      \
  do {                                                \
    if (DBG_ENABLED) {                                \
      std::cerr << "[DBG][SEQ] " << msg << std::endl; \
    }                                                 \
  } while (0)

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

  const double left = p.left;
  const double right = p.right;

  x_.clear();
  y_.clear();

  x_.push_back(left);
  x_.push_back(right);

  const double f_left = p.func(left);
  const double f_right = p.func(right);

  DBG("Init: left=" << left << " right=" << right << " f_left=" << f_left << " f_right=" << f_right);

  if (!std::isfinite(f_left) || !std::isfinite(f_right)) {
    return false;
  }

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

  return true;
}

double SizovDGlobalSearchSEQ::EstimateM(double reliability) const {
  double max_slope = 0.0;

  for (std::size_t i = 1; i < x_.size(); ++i) {
    const double dx = std::abs(x_[i] - x_[i - 1]);
    if (dx > 0.0) {
      const double y_r = y_[i];
      const double y_l = y_[i - 1];

      if (!std::isfinite(y_r) || !std::isfinite(y_l)) {
        continue;
      }

      const double dy = std::abs(y_r - y_l) / dx;
      if (std::isfinite(dy)) {
        max_slope = std::max(max_slope, dy);
      }
    }
  }

  constexpr double kMinimalSlope = 1e-2;
  const double clamped_slope = std::max(max_slope, kMinimalSlope);
  const double m = reliability * clamped_slope;

  DBG("EstimateM: max_slope=" << max_slope << " clamped_slope=" << clamped_slope << " reliability=" << reliability
                              << " M=" << m);

  return m;
}

double SizovDGlobalSearchSEQ::Characteristic(std::size_t idx, double m) const {
  const double x_r = x_[idx];
  const double x_l = x_[idx - 1];
  const double y_r = y_[idx];
  const double y_l = y_[idx - 1];

  const double dx = x_r - x_l;
  const double df = y_r - y_l;

  const double term1 = m * dx;
  const double term2 = (df * df) / (m * dx);
  const double term3 = 2.0 * (y_r + y_l);

  return (term1 + term2) - term3;
}

double SizovDGlobalSearchSEQ::NewPoint(std::size_t idx, double m) const {
  const double x_r = x_[idx];
  const double x_l = x_[idx - 1];
  const double y_r = y_[idx];
  const double y_l = y_[idx - 1];

  const double mid = 0.5 * (x_r + x_l);
  const double shift = (y_r - y_l) / (2.0 * m);

  double x_new = mid - shift;

  if (x_new <= x_l || x_new >= x_r) {
    x_new = mid;
  }

  return x_new;
}

bool SizovDGlobalSearchSEQ::RunImpl() {
  const auto &p = GetInput();

  if (x_.size() < 2) {
    return false;
  }

  for (int iter = 0; iter < p.max_iterations; ++iter) {
    iterations_ = iter + 1;

    const double m = EstimateM(p.reliability);

    double best_char = -std::numeric_limits<double>::infinity();
    std::size_t best_idx = 1;

    for (std::size_t i = 1; i < x_.size(); ++i) {
      const double c = Characteristic(i, m);
      if (c > best_char) {
        best_char = c;
        best_idx = i;
      }
    }

    const double x_l = x_[best_idx - 1];
    const double x_r = x_[best_idx];
    const double width = x_r - x_l;

    DBG("Iter=" << iterations_ << " M=" << m << " best_idx=" << best_idx << " interval=[" << x_l << ", " << x_r << "]"
                << " width=" << width);

    if (width <= p.accuracy) {
      converged_ = true;
      DBG("Stop by accuracy: width=" << width << " accuracy=" << p.accuracy);
      break;
    }

    const double x_new = NewPoint(best_idx, m);
    const double y_new = p.func(x_new);

    DBG("New point: x_new=" << x_new << " y_new=" << y_new);

    if (!std::isfinite(y_new)) {
      DBG("Skip new point: non-finite y_new");
      continue;
    }

    const auto pos = std::ranges::lower_bound(x_, x_new);
    const std::size_t idx = static_cast<std::size_t>(pos - x_.begin());

    x_.insert(pos, x_new);
    y_.insert(y_.begin() + static_cast<std::ptrdiff_t>(idx), y_new);

    if (y_new < best_y_) {
      best_y_ = y_new;
      best_x_ = x_new;
      DBG("Update best: best_x_=" << best_x_ << " best_y_=" << best_y_);
    }
  }

  DBG("RESULT: argmin=" << best_x_ << " value=" << best_y_ << " iterations=" << iterations_
                        << " converged=" << converged_);

  GetOutput() = Solution{
      .argmin = best_x_,
      .value = best_y_,
      .iterations = iterations_,
      .converged = converged_,
  };

  return true;
}

bool SizovDGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_global_search
