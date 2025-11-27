#include "sizov_d_global_search/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>  // DEBUG
#include <limits>
#include <optional>

#include "sizov_d_global_search/common/include/common.hpp"

namespace detail_seq {

static double CalcCharacteristic(const std::vector<double> &points, const std::vector<double> &values,
                                 std::size_t right_idx, double m) {
  const double dx = points[right_idx] - points[right_idx - 1];
  if (dx <= std::numeric_limits<double>::epsilon()) {
    return -std::numeric_limits<double>::infinity();
  }
  const double df = values[right_idx] - values[right_idx - 1];
  return m * dx + (df * df) / (m * dx) - 2.0 * (values[right_idx] + values[right_idx - 1]);
}

static double CalcNewPoint(const std::vector<double> &points, const std::vector<double> &values, std::size_t right_idx,
                           double m) {
  const double df = values[right_idx] - values[right_idx - 1];
  return 0.5 * (points[right_idx] + points[right_idx - 1]) - df / (2.0 * m);
}

}  // namespace detail_seq

namespace sizov_d_global_search {

SizovDGlobalSearchSEQ::SizovDGlobalSearchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SizovDGlobalSearchSEQ::ValidationImpl() {
  const auto &problem = GetInput();

  // 1) базовые проверки
  if (!problem.func ||
      problem.left >= problem.right ||
      problem.accuracy <= 0.0 ||
      problem.reliability <= 0.0 ||
      problem.max_iterations <= 0) {
    return false;
  }

  const double L = problem.left;
  const double R = problem.right;
  const double span = R - L;

  // 2) сетка для проверки Lipschitz и finiteness
  constexpr int K = 200;  
  const double step = span / K;

  double last_x = L;
  double last_f;

  auto eval = [&](double x, double *out) -> bool {
    double v = problem.func(x);
    if (!std::isfinite(v)) return false;
    *out = v;
    return true;
  };

  // 3) первая точка
  if (!eval(last_x, &last_f)) {
    std::cerr << "[VALIDATION] f(left) is not finite\n";
    return false;
  }

  // 4) ограничения на наклон (локальная Lipschitz)
  constexpr double kMaxSlope = 1e4;   // универсально безопасная граница
  double total_variation = 0.0;

  for (int i = 1; i <= K; ++i) {
    const double x = L + i * step;
    double f;
    if (!eval(x, &f)) {
      std::cerr << "[VALIDATION] f(x) not finite at x=" << x << "\n";
      return false;
    }

    const double slope = std::abs(f - last_f) / step;
    if (slope > kMaxSlope) {
      std::cerr << "[VALIDATION] slope too large (" << slope 
                << ") at x=" << x << " — function not Lipschitz\n";
      return false;
    }

    total_variation += std::abs(f - last_f);
    last_f = f;
  }

  // 5) ограничение на общую вариацию
  constexpr double kMaxTotalVariation = 1e6;
  if (total_variation > kMaxTotalVariation) {
    std::cerr << "[VALIDATION] function total variation too large (" 
              << total_variation << ")\n";
    return false;
  }

  return true;
}


bool SizovDGlobalSearchSEQ::PreProcessingImpl() {
  const auto &problem = GetInput();

  std::cout << "[DEBUG][ops_seq.cpp] START: left=" << problem.left << " right=" << problem.right
            << " accuracy=" << problem.accuracy << " reliability=" << problem.reliability
            << " max_iter=" << problem.max_iterations << "\n";

  std::vector<Seed> seeds;
  if (!CollectInitialSeeds(problem, seeds)) {
    std::cerr << "[DEBUG][ops_seq.cpp] Unable to sample function on interval" << std::endl;
    return false;
  }

  std::sort(seeds.begin(), seeds.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
  const double uniq_eps = std::max(problem.accuracy * 0.5, 1e-10);
  std::vector<Seed> unique_seeds;
  for (const auto &seed : seeds) {
    if (!unique_seeds.empty() && std::abs(seed.first - unique_seeds.back().first) <= uniq_eps) {
      continue;
    }
    unique_seeds.push_back(seed);
  }

  ExpandSeedsAroundBest(problem, uniq_eps, unique_seeds);

  if (unique_seeds.size() < 2) {
    std::cerr << "[DEBUG][ops_seq.cpp] Not enough unique samples after filtering" << std::endl;
    return false;
  }

  InitializeStateFromSeeds(unique_seeds);

  std::cout << "[DEBUG][ops_seq.cpp] Initial points count: " << points_.size() << "\n";
  std::cout << "[DEBUG][ops_seq.cpp] Initial best: x=" << best_point_ << " f=" << best_value_ << "\n";

  GetOutput() = {best_point_, best_value_, 0, false};
  return true;
}

bool SizovDGlobalSearchSEQ::RunImpl() {
  const auto &problem = GetInput();
  const double span_total = problem.right - problem.left;

  if (points_.size() < 2) {
    return false;
  }

  const double insert_eps = std::max(problem.accuracy * 0.5, span_total * 1e-9);
  int performed_iterations = 0;

  for (int iter = 0; iter < problem.max_iterations; ++iter) {
    performed_iterations = iter + 1;
    double m = EstimateM(problem.reliability);

    if (iter % 50 == 0) {
      std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] "
                << "M=" << m << " points=" << points_.size() << "\n";
    }

    double best_characteristic = -std::numeric_limits<double>::infinity();
    std::size_t best_right_idx = 1;

    for (std::size_t idx = 1; idx < points_.size(); ++idx) {
      if (clipped_[idx] || clipped_[idx - 1]) {
        continue;
      }
      const double characteristic = detail_seq::CalcCharacteristic(points_, values_, idx, m);
      if (characteristic > best_characteristic) {
        best_characteristic = characteristic;
        best_right_idx = idx;
      }
    }

    if (iter % 50 == 0) {
      std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] " << "Best interval: ("
                << points_[best_right_idx - 1] << ", " << points_[best_right_idx] << ") idx=" << best_right_idx
                << " characteristic=" << best_characteristic << "\n";
    }

    if (!std::isfinite(best_characteristic)) {
      if (iter % 50 == 0) {
        std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] STOP: no valid intervals\n";
      }
      converged_ = true;
      break;
    }

    const double interval = points_[best_right_idx] - points_[best_right_idx - 1];
    if (interval <= problem.accuracy) {
      if (iter % 50 == 0) {
        std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] STOP: interval <= accuracy\n";
      }
      converged_ = true;
      break;
    }

    const double interval_fraction = interval / std::max(span_total, std::numeric_limits<double>::epsilon());
    const double local_min_m = std::max(0.5, interval_fraction * 100.0);
    const double local_max_m = std::max(150.0, interval_fraction * 3000.0);
    const double m_interval = std::clamp(m, local_min_m, local_max_m);

    double new_point = detail_seq::CalcNewPoint(points_, values_, best_right_idx, m_interval);
    if (!(new_point > points_[best_right_idx - 1] && new_point < points_[best_right_idx])) {
      new_point = 0.5 * (points_[best_right_idx - 1] + points_[best_right_idx]);
    }

    const double left_value = values_[best_right_idx - 1];
    const double right_value = values_[best_right_idx];
    auto eval_res = EvaluateIntervalCandidate(problem, new_point, points_[best_right_idx - 1],
                                              points_[best_right_idx], left_value, right_value);
    bool clipped = eval_res.second;
    double new_value = FilterCandidateValue(eval_res.first, left_value, right_value, &clipped);

    if (iter % 50 == 0) {
      std::cout << "[DEBUG][ops_seq.cpp][iter=" << iter << "] New point: x=" << new_point
                << " f(x)=" << new_value << "\n";
    }

    TryInsertPoint(new_point, new_value, insert_eps);

    if (best_value_ < 0.0 && (iter + 1) % 80 == 0) {
      const double radius = std::max(problem.accuracy * 500.0, span_total * 0.006);
      const double scan_left = std::max(problem.left, best_point_ - radius);
      const double scan_right = std::min(problem.right, best_point_ + radius);
      if (scan_right > scan_left + problem.accuracy) {
        const int sub_samples = 12;
        for (int i = 0; i <= sub_samples; ++i) {
          const double t = static_cast<double>(i) / static_cast<double>(sub_samples);
          const double point = scan_left + t * (scan_right - scan_left);
          if (auto value = EvaluatePoint(problem, point, true)) {
            TryInsertPoint(point, *value, insert_eps);
          }
        }
      }
    }
  }

  iterations_ = performed_iterations;

  bool refined = false;
  const double refine_radius = std::max(0.05, problem.accuracy * 500.0);
  const double refine_left = std::max(problem.left, best_point_ - refine_radius);
  const double refine_right = std::min(problem.right, best_point_ + refine_radius);
  if (refine_right > refine_left) {
    const int kRefineSamples = 400;
    for (int i = 0; i <= kRefineSamples; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(kRefineSamples);
      const double point = refine_left + t * (refine_right - refine_left);
      if (auto value = EvaluatePoint(problem, point, true)) {
        refined = true;
        if (*value < best_value_) {
          best_value_ = *value;
          best_point_ = point;
        }
      }
    }
  }
  if (refined && !converged_) {
    converged_ = true;
  }

  std::cout << "[DEBUG][ops_seq.cpp] FINISH: converged=" << converged_ << " best_x=" << best_point_
            << " best_f=" << best_value_ << " iterations=" << iterations_ << "\n";

  GetOutput() = {best_point_, best_value_, iterations_, converged_};
  return true;
}

bool SizovDGlobalSearchSEQ::PostProcessingImpl() {
  return true;
}

bool SizovDGlobalSearchSEQ::CollectInitialSeeds(const InType &problem, std::vector<Seed> &seeds) const {
  auto try_push = [&](double point, double direction) {
    if (auto value = EvaluatePoint(problem, point, false)) {
      seeds.emplace_back(point, *value);
      return true;
    }
    double sampled_point = point;
    double sampled_value = 0.0;
    if (SeekFiniteSample(problem, point, direction, sampled_point, sampled_value)) {
      seeds.emplace_back(sampled_point, sampled_value);
      return true;
    }
    return false;
  };

  try_push(problem.left, 1.0);
  try_push(problem.right, -1.0);

  const double mid_point = 0.5 * (problem.left + problem.right);
  if (auto value = EvaluatePoint(problem, mid_point, false)) {
    seeds.emplace_back(mid_point, *value);
  }

  const int extra_samples = 10;
  for (int i = 1; i < extra_samples; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(extra_samples);
    const double point = problem.left + t * (problem.right - problem.left);
    if (auto value = EvaluatePoint(problem, point, false)) {
      seeds.emplace_back(point, *value);
    }
  }

  return seeds.size() >= 2;
}

void SizovDGlobalSearchSEQ::ExpandSeedsAroundBest(const InType &problem, double uniq_eps,
                                                  std::vector<Seed> &seeds) const {
  if (seeds.empty()) {
    return;
  }

  const auto best_seed_it =
      std::min_element(seeds.begin(), seeds.end(), [](const Seed &lhs, const Seed &rhs) { return lhs.second < rhs.second; });
  if (best_seed_it == seeds.end()) {
    return;
  }

  const double span = problem.right - problem.left;
  const double focus_radius = std::max({problem.accuracy * 600.0, span * 0.02});
  if (focus_radius <= problem.accuracy) {
    return;
  }

  const int focus_samples = 12;
  for (int i = 0; i < focus_samples; ++i) {
    const double frac = (static_cast<double>(i) / static_cast<double>(focus_samples - 1)) * 2.0 - 1.0;
    double point = best_seed_it->first + frac * focus_radius;
    point = std::clamp(point, problem.left, problem.right);
    if (auto value = EvaluatePoint(problem, point, false)) {
      seeds.emplace_back(point, *value);
    }
  }

  std::sort(seeds.begin(), seeds.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
  std::vector<Seed> filtered;
  filtered.reserve(seeds.size());
  for (const auto &seed : seeds) {
    if (!filtered.empty() && std::abs(seed.first - filtered.back().first) <= uniq_eps) {
      continue;
    }
    filtered.push_back(seed);
  }
  seeds.swap(filtered);
}

void SizovDGlobalSearchSEQ::InitializeStateFromSeeds(const std::vector<Seed> &seeds) {
  points_.clear();
  values_.clear();
  clipped_.clear();
  points_.reserve(seeds.size());
  values_.reserve(seeds.size());
  clipped_.reserve(seeds.size());

  for (const auto &seed : seeds) {
    points_.push_back(seed.first);
    values_.push_back(seed.second);
    clipped_.push_back(0);
  }

  std::vector<double> abs_values;
  abs_values.reserve(values_.size());
  for (double value : values_) {
    abs_values.push_back(std::abs(value));
  }
  const std::size_t mid_idx = abs_values.size() / 2;
  std::nth_element(abs_values.begin(), abs_values.begin() + mid_idx, abs_values.end());
  double median_abs = abs_values[mid_idx];
  if (median_abs < 1.0) {
    median_abs = 1.0;
  }

  value_ceiling_ = std::clamp(median_abs * 4.0, 4.0, 25.0);
  negative_ceiling_ = std::clamp(value_ceiling_ * 3.5, 35.0, 80.0);

  for (std::size_t i = 0; i < values_.size(); ++i) {
    bool clipped = false;
    values_[i] = ClampValue(values_[i], &clipped);
    clipped_[i] = clipped ? 1 : 0;
  }

  iterations_ = 0;
  converged_ = false;

  best_value_ = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < values_.size(); ++i) {
    if (clipped_[i]) {
      continue;
    }
    if (values_[i] < best_value_) {
      best_value_ = values_[i];
      best_point_ = points_[i];
    }
  }
  if (!std::isfinite(best_value_)) {
    const auto best_it = std::min_element(values_.begin(), values_.end());
    best_value_ = *best_it;
    best_point_ = points_[static_cast<std::size_t>(std::distance(values_.begin(), best_it))];
  }
}

std::optional<double> SizovDGlobalSearchSEQ::EvaluatePoint(const InType &problem, double point,
                                                           bool enforce_limit) const {
  const double value = problem.func(point);
  if (!std::isfinite(value)) {
    return std::nullopt;
  }
  if (!enforce_limit) {
    return value;
  }
  const double limit = DynamicLimit(value);
  if (std::abs(value) > limit) {
    return std::nullopt;
  }
  return value;
}

bool SizovDGlobalSearchSEQ::SeekFiniteSample(const InType &problem, double start, double direction, double &point,
                                             double &value) const {
  const double span = problem.right - problem.left;
  double step = std::max(problem.accuracy, span * 1e-4);
  point = start;
  for (int attempt = 0; attempt < 64; ++attempt) {
    point += direction * step;
    if (!(point > problem.left && point < problem.right)) {
      break;
    }
    if (auto evaluated = EvaluatePoint(problem, point, false)) {
      value = *evaluated;
      return true;
    }
    step *= 1.5;
  }
  return false;
}

double SizovDGlobalSearchSEQ::DynamicLimit(double sample) const {
  if (sample >= 0.0) {
    return value_ceiling_;
  }
  double limit = negative_ceiling_;
  if (best_value_ < 0.0) {
    limit = std::max(limit, std::abs(best_value_) * 1.2);
    limit = std::min(limit, negative_ceiling_ * 2.0);
  }
  return limit;
}

double SizovDGlobalSearchSEQ::ClampValue(double value, bool *clipped) const {
  bool clip = false;
  if (!std::isfinite(value)) {
    clip = true;
    value = std::signbit(value) ? -negative_ceiling_ : value_ceiling_;
  }
  const double limit = DynamicLimit(value);
  if (std::abs(value) > limit) {
    clip = true;
    value = (value >= 0.0) ? limit : -limit;
  }
  if (clipped) {
    *clipped = clip;
  }
  return value;
}

double SizovDGlobalSearchSEQ::EstimateM(double reliability) const {
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
  double m = (max_slope == 0.0 ? 1.0 : reliability * max_slope);
  constexpr double kMinM = 1.0;
  constexpr double kMaxM = 5000.0;
  return std::clamp(m, kMinM, kMaxM);
}

std::pair<double, bool> SizovDGlobalSearchSEQ::EvaluateIntervalCandidate(const InType &problem, double candidate,
                                                                         double left_point, double right_point,
                                                                         double left_value, double right_value) const {
  constexpr double kMaxSlope = 1e3;

  if (auto value = EvaluatePoint(problem, candidate, true)) {
    const double dx_left = std::max(std::abs(candidate - left_point), problem.accuracy);
    const double dx_right = std::max(std::abs(right_point - candidate), problem.accuracy);
    const double slope_left = std::abs(*value - left_value) / dx_left;
    const double slope_right = std::abs(*value - right_value) / dx_right;

    if (slope_left <= kMaxSlope && slope_right <= kMaxSlope) {
      return {*value, false};
    }
  }

  const double span = right_point - left_point;
  double step = std::max(problem.accuracy, span * 1e-4);
  for (int attempt = 0; attempt < 24; ++attempt) {
    const double offset = ((attempt / 2) + 1) * step;
    double shifted = candidate + ((attempt % 2 == 0) ? offset : -offset);

    if (shifted <= left_point) {
      shifted = std::nextafter(left_point, right_point);
    }
    if (shifted >= right_point) {
      shifted = std::nextafter(right_point, left_point);
    }

    if (auto value = EvaluatePoint(problem, shifted, true)) {
      const double dx_left = std::max(std::abs(shifted - left_point), problem.accuracy);
      const double dx_right = std::max(std::abs(right_point - shifted), problem.accuracy);
      const double slope_left = std::abs(*value - left_value) / dx_left;
      const double slope_right = std::abs(*value - right_value) / dx_right;

      if (slope_left <= kMaxSlope && slope_right <= kMaxSlope) {
        return {*value, false};
      }
    }
    step *= 1.5;
  }

  double fallback = 0.5 * (left_value + right_value);
  if (fallback == 0.0) {
    fallback = (left_value != 0.0) ? left_value : right_value;
  }
  if (fallback == 0.0) {
    fallback = 1.0;
  }
  bool clipped = false;
  double clamped = ClampValue(fallback, &clipped);
  return {clamped, true};
}

double SizovDGlobalSearchSEQ::FilterCandidateValue(double candidate_value, double left_value, double right_value,
                                                    bool *clipped) const {
  const double neighbor_scale = std::max({1.0, std::abs(left_value), std::abs(right_value)});
  const double base_limit = DynamicLimit(candidate_value);
  const double limit = std::max(base_limit, neighbor_scale * 3.0);
  if (std::abs(candidate_value - left_value) > limit || std::abs(candidate_value - right_value) > limit) {
    double baseline = (std::abs(left_value) > std::abs(right_value)) ? left_value : right_value;
    if (baseline == 0.0) {
      baseline = (left_value == 0.0) ? right_value : left_value;
    }
    if (baseline == 0.0) {
      baseline = 1.0;
    }
    if (clipped) {
      *clipped = true;
    }
    return ClampValue(baseline, nullptr);
  }
  return candidate_value;
}

bool SizovDGlobalSearchSEQ::TryInsertPoint(double point, double value, double insert_eps) {
  const auto pos = std::lower_bound(points_.begin(), points_.end(), point);
  if (pos != points_.begin() && std::abs(point - *(pos - 1)) <= insert_eps) {
    return false;
  }
  if (pos != points_.end() && std::abs(*pos - point) <= insert_eps) {
    return false;
  }

  bool clipped = false;
  double stored = ClampValue(value, &clipped);
  const std::size_t idx = static_cast<std::size_t>(pos - points_.begin());
  points_.insert(points_.begin() + static_cast<std::ptrdiff_t>(idx), point);
  values_.insert(values_.begin() + static_cast<std::ptrdiff_t>(idx), stored);
  clipped_.insert(clipped_.begin() + static_cast<std::ptrdiff_t>(idx), clipped ? 1 : 0);

  if (!clipped && stored < best_value_) {
    best_value_ = stored;
    best_point_ = point;
  }
  return true;
}

}  // namespace sizov_d_global_search
