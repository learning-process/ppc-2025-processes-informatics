#include "kondakov_v_global_search/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <vector>

#include "kondakov_v_global_search/common/include/common.hpp"

namespace kondakov_v_global_search {

KondakovVGlobalSearchSEQ::KondakovVGlobalSearchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KondakovVGlobalSearchSEQ::ValidationImpl() {
  const auto &cfg = GetInput();
  if (!cfg.func) {
    return false;
  }
  if (cfg.left >= cfg.right) {
    return false;
  }
  if (cfg.accuracy <= 0.0) {
    return false;
  }
  if (cfg.reliability <= 0.0) {
    return false;
  }
  if (cfg.max_iterations <= 0) {
    return false;
  }
  return true;
}

bool KondakovVGlobalSearchSEQ::PreProcessingImpl() {
  const auto &cfg = GetInput();

  points_x_.clear();
  values_y_.clear();
  points_x_.reserve(cfg.max_iterations + 10);
  values_y_.reserve(cfg.max_iterations + 10);

  double f_a = cfg.func(cfg.left);
  double f_b = cfg.func(cfg.right);

  if (!std::isfinite(f_a) || !std::isfinite(f_b)) {
    return false;
  }

  points_x_.push_back(cfg.left);
  points_x_.push_back(cfg.right);
  values_y_.push_back(f_a);
  values_y_.push_back(f_b);

  if (f_a < f_b) {
    best_point_ = cfg.left;
    best_value_ = f_a;
  } else {
    best_point_ = cfg.right;
    best_value_ = f_b;
  }

  total_evals_ = 0;
  has_converged_ = false;
  return true;
}

double KondakovVGlobalSearchSEQ::ComputeAdaptiveLipschitzEstimate(double r) const {
  const double min_allowed_slope = 1e-2;
  if (points_x_.size() < 2) {
    return r * min_allowed_slope;
  }

  double max_local_slope = min_allowed_slope;
  for (std::size_t i = 1; i < points_x_.size(); ++i) {
    double dx = points_x_[i] - points_x_[i - 1];
    if (dx <= 0.0) {
      continue;
    }
    double dy = std::abs(values_y_[i] - values_y_[i - 1]);
    if (!std::isfinite(dy)) {
      continue;
    }
    double slope = dy / dx;
    if (std::isfinite(slope) && slope > max_local_slope) {
      max_local_slope = slope;
    }
  }
  return r * max_local_slope;
}

double KondakovVGlobalSearchSEQ::IntervalMerit(std::size_t i, double l_est) const {
  double x_l = points_x_[i - 1];
  double x_r = points_x_[i];
  double f_l = values_y_[i - 1];
  double f_r = values_y_[i];
  double h = x_r - x_l;
  double df = f_r - f_l;
  return (l_est * h) - (2.0 * (f_l + f_r)) + ((df * df) / (l_est * h));
}

double KondakovVGlobalSearchSEQ::ProposeTrialPoint(std::size_t i, double l_est) const {
  double x_l = points_x_[i - 1];
  double x_r = points_x_[i];
  double f_l = values_y_[i - 1];
  double f_r = values_y_[i];
  double midpoint = 0.5 * (x_l + x_r);
  double asymmetry = (f_r - f_l) / (2.0 * l_est);
  double candidate = midpoint - asymmetry;
  if (candidate <= x_l || candidate >= x_r) {
    candidate = midpoint;
  }
  return candidate;
}

std::size_t KondakovVGlobalSearchSEQ::LocateInsertionIndex(double x) const {
  auto it = std::ranges::lower_bound(points_x_, x);
  return static_cast<std::size_t>(std::distance(points_x_.begin(), it));
}

void KondakovVGlobalSearchSEQ::InsertEvaluation(double x, double fx) {
  std::size_t idx = LocateInsertionIndex(x);
  auto pos = static_cast<std::vector<double>::difference_type>(idx);
  points_x_.insert(points_x_.begin() + pos, x);
  values_y_.insert(values_y_.begin() + pos, fx);
  if (fx < best_value_) {
    best_value_ = fx;
    best_point_ = x;
  }
}

bool KondakovVGlobalSearchSEQ::RunImpl() {
  const auto &cfg = GetInput();
  if (points_x_.size() < 2) {
    return false;
  }

  double lipschitz_estimate = ComputeAdaptiveLipschitzEstimate(cfg.reliability);

  for (int step = 0; step < cfg.max_iterations; ++step) {
    total_evals_ = step + 1;
    if ((step % 10) == 0) {
      lipschitz_estimate = ComputeAdaptiveLipschitzEstimate(cfg.reliability);
    }
    if (points_x_.size() < 2) {
      has_converged_ = false;
      break;
    }

    std::size_t best_interval = 1;
    double max_merit = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < points_x_.size(); ++i) {
      double merit = IntervalMerit(i, lipschitz_estimate);
      if (merit > max_merit) {
        max_merit = merit;
        best_interval = i;
      }
    }

    double interval_width = points_x_[best_interval] - points_x_[best_interval - 1];
    if (interval_width <= cfg.accuracy) {
      has_converged_ = true;
      break;
    }

    double trial_x = ProposeTrialPoint(best_interval, lipschitz_estimate);
    double trial_f = cfg.func(trial_x);
    if (!std::isfinite(trial_f)) {
      continue;
    }

    InsertEvaluation(trial_x, trial_f);
  }

  GetOutput() =
      Solution{.argmin = best_point_, .value = best_value_, .iterations = total_evals_, .converged = has_converged_};
  return true;
}

bool KondakovVGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kondakov_v_global_search
