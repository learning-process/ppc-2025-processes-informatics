#include "sizov_d_global_search/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"

namespace {

// -------------------- Strongin characteristic --------------------

double calc_characteristic(const std::vector<double> &points, const std::vector<double> &values, std::size_t r,
                           double m) {
  const double dx = points[r] - points[r - 1];
  if (dx <= std::numeric_limits<double>::epsilon()) {
    return -std::numeric_limits<double>::infinity();
  }
  const double df = values[r] - values[r - 1];
  const double term1 = m * dx;
  const double term2 = (df * df) / (m * dx);
  const double term3 = 2.0 * (values[r] + values[r - 1]);
  return term1 + term2 - term3;
}

double calc_new_point(const std::vector<double> &points, const std::vector<double> &values, std::size_t r, double m) {
  const double df = values[r] - values[r - 1];
  const double mid = 0.5 * (points[r] + points[r - 1]);
  return mid - df / (2.0 * m);
}

}  // namespace

namespace sizov_d_global_search {

// -------------------- Конструктор --------------------

SizovDGlobalSearchSEQ::SizovDGlobalSearchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

// -------------------- Validation --------------------

bool SizovDGlobalSearchSEQ::ValidationImpl() {
  const auto &p = GetInput();
  if (!p.func || p.left >= p.right || p.accuracy <= 0.0 || p.reliability <= 0.0 || p.max_iterations <= 0) {
    return false;
  }

  constexpr int kGrid = 200;
  const double L = p.left;
  const double R = p.right;
  const double span = R - L;
  const double step = span / kGrid;
  constexpr double kMaxSlope = 1e4;
  constexpr double kMaxVariation = 1e6;

  double last_f{};
  if (!std::isfinite(p.func(L))) {
    return false;
  }
  last_f = p.func(L);

  double total = 0.0;

  for (int i = 1; i <= kGrid; ++i) {
    const double x = L + i * step;
    const double f = p.func(x);
    if (!std::isfinite(f)) {
      return false;
    }
    const double slope = std::abs(f - last_f) / step;
    if (slope > kMaxSlope) {
      return false;
    }
    total += std::abs(f - last_f);
    last_f = f;
  }

  return total <= kMaxVariation;
}

// -------------------- PreProcessing --------------------

bool SizovDGlobalSearchSEQ::PreProcessingImpl() {
  const auto &p = GetInput();

  std::vector<Seed> seeds;
  if (!collect_initial_seeds(p, seeds)) {
    return false;
  }

  std::ranges::sort(seeds, {}, &Seed::first);

  const double uniq = std::max(p.accuracy * 0.5, 1e-10);

  std::vector<Seed> uniq_seeds;
  uniq_seeds.reserve(seeds.size());

  for (const auto &s : seeds) {
    if (!uniq_seeds.empty() && std::abs(s.first - uniq_seeds.back().first) <= uniq) {
      continue;
    }
    uniq_seeds.push_back(s);
  }

  expand_seeds_around_best(p, uniq, uniq_seeds);

  if (uniq_seeds.size() < 2) {
    return false;
  }

  initialize_state_from_seeds(uniq_seeds);

  GetOutput() = Solution{best_point_, best_value_, 0, false};
  return true;
}

// -------------------- Run --------------------

bool SizovDGlobalSearchSEQ::RunImpl() {
  const auto &p = GetInput();
  const double span_total = p.right - p.left;

  if (points_.size() < 2) {
    return false;
  }

  const double insert_eps = std::max(p.accuracy * 0.5, span_total * 1e-9);

  for (int iter = 0; iter < p.max_iterations; ++iter) {
    iterations_ = iter + 1;

    const double m = estimate_m(p.reliability);

    // 1. Найти лучший интервал
    double best_char = -std::numeric_limits<double>::infinity();
    std::size_t br = 1;

    for (std::size_t i = 1; i < points_.size(); ++i) {
      if (clipped_[i] || clipped_[i - 1]) {
        continue;
      }
      double c = calc_characteristic(points_, values_, i, m);
      if (c > best_char) {
        best_char = c;
        br = i;
      }
    }

    if (!std::isfinite(best_char)) {
      converged_ = true;
      break;
    }

    // 2. Проверка на точность
    const double interval = points_[br] - points_[br - 1];
    if (interval <= p.accuracy) {
      converged_ = true;
      break;
    }

    // 3. Выбрать точку внутри интервала
    const double f = interval / std::max(span_total, std::numeric_limits<double>::epsilon());
    const double m_local = std::clamp(m, std::max(0.5, f * 100.0), std::max(150.0, f * 3000.0));

    double x_new = calc_new_point(points_, values_, br, m_local);
    if (x_new <= points_[br - 1] || x_new >= points_[br]) {
      x_new = 0.5 * (points_[br] + points_[br - 1]);
    }

    const double lv = values_[br - 1];
    const double rv = values_[br];

    auto [raw, clipped] = eval_interval_candidate(p, x_new, points_[br - 1], points_[br], lv, rv);

    double y_new = filter_candidate_value(raw, lv, rv, &clipped);

    try_insert_point(x_new, y_new, insert_eps);
  }

  // -------------------- Финишная локальная refinement --------------------

  const double rad = std::max(0.05, p.accuracy * 500.0);
  const double L = std::max(p.left, best_point_ - rad);
  const double R = std::min(p.right, best_point_ + rad);

  if (R > L) {
    constexpr int kSamples = 400;
    for (int i = 0; i <= kSamples; ++i) {
      double t = static_cast<double>(i) / kSamples;
      double x = L + t * (R - L);
      if (auto v = evaluate_point(p, x, true)) {
        if (*v < best_value_) {
          best_value_ = *v;
          best_point_ = x;
        }
      }
    }
    converged_ = true;
  }

  GetOutput() = Solution{best_point_, best_value_, iterations_, converged_};
  return true;
}

// -------------------- PostProcessing --------------------

bool SizovDGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

// -------------------- Helpers --------------------

bool SizovDGlobalSearchSEQ::collect_initial_seeds(const InType &problem, std::vector<Seed> &seeds) const {
  auto try_push = [&](double x, double dir) {
    if (auto v = evaluate_point(problem, x, false)) {
      seeds.emplace_back(x, *v);
      return true;
    }
    if (auto s = seek_finite_sample(problem, x, dir)) {
      seeds.emplace_back(s->point, s->value);
      return true;
    }
    return false;
  };

  try_push(problem.left, +1.0);
  try_push(problem.right, -1.0);

  // mid
  const double mid = 0.5 * (problem.left + problem.right);
  if (auto v = evaluate_point(problem, mid, false)) {
    seeds.emplace_back(mid, *v);
  }

  // uniform samples
  constexpr int kExtra = 10;
  for (int i = 1; i < kExtra; ++i) {
    const double t = static_cast<double>(i) / kExtra;
    const double x = problem.left + t * (problem.right - problem.left);
    if (auto v = evaluate_point(problem, x, false)) {
      seeds.emplace_back(x, *v);
    }
  }

  return seeds.size() >= 2;
}

void SizovDGlobalSearchSEQ::expand_seeds_around_best(const InType &problem, double uniq_eps,
                                                     std::vector<Seed> &seeds) const {
  if (seeds.empty()) {
    return;
  }

  const auto it = std::ranges::min_element(seeds, {}, &Seed::second);
  if (it == seeds.end()) {
    return;
  }

  const double span = problem.right - problem.left;
  const double rad = std::max(problem.accuracy * 600.0, span * 0.02);
  if (rad <= problem.accuracy) {
    return;
  }

  constexpr int kSamples = 12;
  for (int i = 0; i < kSamples; ++i) {
    double frac = (static_cast<double>(i) / (kSamples - 1)) * 2.0 - 1.0;
    double x = it->first + frac * rad;
    x = std::clamp(x, problem.left, problem.right);
    if (auto v = evaluate_point(problem, x, false)) {
      seeds.emplace_back(x, *v);
    }
  }

  std::ranges::sort(seeds, {}, &Seed::first);

  std::vector<Seed> tmp;
  tmp.reserve(seeds.size());
  for (const auto &s : seeds) {
    if (!tmp.empty() && std::abs(s.first - tmp.back().first) <= uniq_eps) {
      continue;
    }
    tmp.push_back(s);
  }
  seeds.swap(tmp);
}

void SizovDGlobalSearchSEQ::initialize_state_from_seeds(const std::vector<Seed> &seeds) {
  points_.clear();
  values_.clear();
  clipped_.clear();

  for (auto [x, y] : seeds) {
    points_.push_back(x);
    values_.push_back(y);
    clipped_.push_back(false);
  }

  std::vector<double> av;
  av.reserve(values_.size());
  for (double v : values_) {
    av.push_back(std::abs(v));
  }

  std::nth_element(av.begin(), av.begin() + av.size() / 2, av.end());
  double med = std::max(av[av.size() / 2], 1.0);

  value_ceiling_ = std::clamp(med * 4.0, 4.0, 25.0);
  negative_ceiling_ = std::clamp(value_ceiling_ * 3.5, 35.0, 80.0);

  // clamp values
  for (std::size_t i = 0; i < values_.size(); ++i) {
    bool c = false;
    values_[i] = clamp_value(values_[i], &c);
    clipped_[i] = c;
  }

  converged_ = false;
  iterations_ = 0;

  best_value_ = std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < values_.size(); ++i) {
    if (!clipped_[i] && values_[i] < best_value_) {
      best_value_ = values_[i];
      best_point_ = points_[i];
    }
  }

  if (!std::isfinite(best_value_)) {
    const auto it = std::ranges::min_element(values_);
    best_value_ = *it;
    best_point_ = points_[std::distance(values_.begin(), it)];
  }
}

std::optional<double> SizovDGlobalSearchSEQ::evaluate_point(const InType &problem, double x, bool enforce) const {
  const double v = problem.func(x);
  if (!std::isfinite(v)) {
    return std::nullopt;
  }
  if (!enforce) {
    return v;
  }

  if (std::abs(v) > dynamic_limit(v)) {
    return std::nullopt;
  }
  return v;
}

std::optional<SizovDGlobalSearchSEQ::SampleResult> SizovDGlobalSearchSEQ::seek_finite_sample(const InType &problem,
                                                                                             double start,
                                                                                             double dir) const {
  const double span = problem.right - problem.left;
  double step = std::max(problem.accuracy, span * 1e-4);

  double x = start;

  for (int i = 0; i < 64; ++i) {
    x += dir * step;

    if (x <= problem.left || x >= problem.right) {
      break;
    }

    if (auto v = evaluate_point(problem, x, false)) {
      return SampleResult{x, *v};
    }

    step *= 1.5;
  }
  return std::nullopt;
}

double SizovDGlobalSearchSEQ::dynamic_limit(double sample) const {
  if (sample >= 0.0) {
    return value_ceiling_;
  }

  double lim = negative_ceiling_;
  if (best_value_ < 0.0) {
    lim = std::max(lim, std::abs(best_value_) * 1.2);
    lim = std::min(lim, negative_ceiling_ * 2.0);
  }
  return lim;
}

double SizovDGlobalSearchSEQ::clamp_value(double v, bool *clipped) const {
  bool c = false;

  if (!std::isfinite(v)) {
    c = true;
    v = std::signbit(v) ? -negative_ceiling_ : value_ceiling_;
  }

  const double lim = dynamic_limit(v);
  if (std::abs(v) > lim) {
    c = true;
    v = (v >= 0.0) ? lim : -lim;
  }

  if (clipped != nullptr) {
    *clipped = c;
  }
  return v;
}

double SizovDGlobalSearchSEQ::estimate_m(double rel) const {
  double max_slope = 0.0;

  for (std::size_t i = 1; i < points_.size(); ++i) {
    if (clipped_[i] || clipped_[i - 1]) {
      continue;
    }
    const double dx = points_[i] - points_[i - 1];
    if (dx <= 0.0) {
      continue;
    }

    const double slope = std::abs(values_[i] - values_[i - 1]) / dx;
    max_slope = std::max(max_slope, slope);
  }

  double m = (max_slope == 0.0 ? 1.0 : rel * max_slope);
  return std::clamp(m, 1.0, 5000.0);
}

std::pair<double, bool> SizovDGlobalSearchSEQ::eval_interval_candidate(const InType &p, double x, double L, double R,
                                                                       double lv, double rv) const {
  constexpr double kMaxSlope = 1e3;

  if (auto v = evaluate_point(p, x, true)) {
    double dxL = std::max(std::abs(x - L), p.accuracy);
    double dxR = std::max(std::abs(R - x), p.accuracy);

    if (std::abs(*v - lv) / dxL <= kMaxSlope && std::abs(*v - rv) / dxR <= kMaxSlope) {
      return {*v, false};
    }
  }

  const double span = R - L;
  double step = std::max(p.accuracy, span * 1e-4);

  for (int a = 0; a < 24; ++a) {
    const double off = (a / 2.0 + 1.0) * step;
    double x2 = x + ((a % 2 == 0) ? off : -off);

    if (x2 <= L) {
      x2 = std::nextafter(L, R);
    }
    if (x2 >= R) {
      x2 = std::nextafter(R, L);
    }

    if (auto v = evaluate_point(p, x2, true)) {
      double dxL = std::max(std::abs(x2 - L), p.accuracy);
      double dxR = std::max(std::abs(R - x2), p.accuracy);
      if (std::abs(*v - lv) / dxL <= kMaxSlope && std::abs(*v - rv) / dxR <= kMaxSlope) {
        return {*v, false};
      }
    }
    step *= 1.5;
  }

  double fb = 0.5 * (lv + rv);
  if (fb == 0.0) {
    fb = (lv != 0.0 ? lv : rv);
  }
  if (fb == 0.0) {
    fb = 1.0;
  }

  bool clipped = false;
  return {clamp_value(fb, &clipped), true};
}

double SizovDGlobalSearchSEQ::filter_candidate_value(double v, double lv, double rv, bool *clipped) const {
  const double s = std::max({1.0, std::abs(lv), std::abs(rv)});
  const double lim = std::max(dynamic_limit(v), s * 3.0);

  if (std::abs(v - lv) > lim || std::abs(v - rv) > lim) {
    double b = (std::abs(lv) > std::abs(rv)) ? lv : rv;
    if (b == 0.0) {
      b = (lv == 0.0 ? rv : lv);
    }
    if (b == 0.0) {
      b = 1.0;
    }

    if (clipped != nullptr) {
      *clipped = true;
    }
    return clamp_value(b, nullptr);
  }

  return v;
}

bool SizovDGlobalSearchSEQ::try_insert_point(double x, double v, double eps) {
  const auto pos = std::ranges::lower_bound(points_, x);

  if (pos != points_.begin() && std::abs(x - *(pos - 1)) <= eps) {
    return false;
  }
  if (pos != points_.end() && std::abs(*pos - x) <= eps) {
    return false;
  }

  bool clipped = false;
  double vv = clamp_value(v, &clipped);

  std::size_t idx = std::distance(points_.begin(), pos);

  points_.insert(points_.begin() + idx, x);
  values_.insert(values_.begin() + idx, vv);
  clipped_.insert(clipped_.begin() + idx, clipped);

  if (!clipped && vv < best_value_) {
    best_value_ = vv;
    best_point_ = x;
  }

  return true;
}

}  // namespace sizov_d_global_search
