#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"

namespace {

constexpr int kValidationGridSize = 200;
constexpr double kMaxValidationSlope = 1e4;
constexpr double kMaxTotalVariation = 1e6;

constexpr int kMaxSeekAttempts = 64;
constexpr int kMaxShiftAttempts = 24;

constexpr double kMinM = 1.0;
constexpr double kMaxM = 5000.0;

constexpr int kLocalScanPeriod = 80;
constexpr int kLocalScanSamples = 12;

constexpr int kRefineSamples = 400;
constexpr double kMinRefineRadius = 0.05;
constexpr double kMinInsertEps = 1e-10;

struct IntervalCharacteristic {
  double characteristic = -std::numeric_limits<double>::infinity();
  int right_index = 1;
};

double Midpoint(double a, double b) { return 0.5 * (a + b); }

double DistanceOrAccuracy(double dx, double accuracy) { return std::max(std::abs(dx), accuracy); }

bool SlopesAreAcceptable(double value, double left_value, double right_value, double point, double left_point,
                         double right_point, double accuracy, double max_slope) {
  const double dx_left = DistanceOrAccuracy(point - left_point, accuracy);
  const double dx_right = DistanceOrAccuracy(right_point - point, accuracy);

  const double slope_left = std::abs(value - left_value) / dx_left;
  const double slope_right = std::abs(value - right_value) / dx_right;
  return slope_left <= max_slope && slope_right <= max_slope;
}

double ComputeFallbackValue(double left_value, double right_value) {
  double fallback = Midpoint(left_value, right_value);
  if (fallback == 0.0) {
    fallback = (left_value != 0.0) ? left_value : right_value;
  }
  if (fallback == 0.0) {
    fallback = 1.0;
  }
  return fallback;
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

namespace detail_mpi {

double CalcCharacteristic(const std::vector<double> &points, const std::vector<double> &values, std::size_t right_idx,
                          double m);
double CalcNewPoint(const std::vector<double> &points, const std::vector<double> &values, std::size_t right_idx,
                    double m);

double CalcCharacteristic(const std::vector<double> &points, const std::vector<double> &values, std::size_t right_idx,
                          double m) {
  if (right_idx == 0 || right_idx >= points.size()) {
    return -std::numeric_limits<double>::infinity();
  }

  const double dx = points[right_idx] - points[right_idx - 1];
  if (dx <= std::numeric_limits<double>::epsilon()) {
    return -std::numeric_limits<double>::infinity();
  }
  const double df = values[right_idx] - values[right_idx - 1];
  const double denom = m * dx;
  if (!(denom > 0.0) || !std::isfinite(denom)) {
    return -std::numeric_limits<double>::infinity();
  }

  const double result = m * dx + (df * df) / denom - 2.0 * (values[right_idx] + values[right_idx - 1]);
  if (!std::isfinite(result)) {
    return -std::numeric_limits<double>::infinity();
  }
  return result;
}

double CalcNewPoint(const std::vector<double> &points, const std::vector<double> &values, std::size_t right_idx,
                    double m) {
  const double df = values[right_idx] - values[right_idx - 1];
  const double mid = Midpoint(points[right_idx], points[right_idx - 1]);
  const double shift = df / (2.0 * m);
  const double candidate = mid - shift;
  if (!std::isfinite(candidate)) {
    return mid;
  }
  return candidate;
}

}  // namespace detail_mpi

namespace sizov_d_global_search {

SizovDGlobalSearchMPI::SizovDGlobalSearchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SizovDGlobalSearchMPI::ValidationImpl() {
  const auto &problem = GetInput();

  if (!problem.func || problem.left >= problem.right || problem.accuracy <= 0.0 || problem.reliability <= 0.0 ||
      problem.max_iterations <= 0) {
    return false;
  }

  const double left = problem.left;
  const double right = problem.right;
  const double span = right - left;

  const double step = span / static_cast<double>(kValidationGridSize);

  auto eval = [&](double x, double *out) -> bool {
    const double v = problem.func(x);
    if (!std::isfinite(v)) {
      return false;
    }
    *out = v;
    return true;
  };

  double last_x = left;
  double last_f = 0.0;
  if (!eval(last_x, &last_f)) {
    std::cerr << "[VALIDATION][MPI] f(left) is not finite\n";
    return false;
  }

  double total_variation = 0.0;

  for (int i = 1; i <= kValidationGridSize; ++i) {
    const double x = left + (static_cast<double>(i) * step);
    double f = 0.0;
    if (!eval(x, &f)) {
      std::cerr << "[VALIDATION][MPI] f(x) not finite at x=" << x << "\n";
      return false;
    }

    const double slope = std::abs(f - last_f) / step;
    if (slope > kMaxValidationSlope) {
      std::cerr << "[VALIDATION][MPI] slope too large at x=" << x << "\n";
      return false;
    }

    total_variation += std::abs(f - last_f);
    last_f = f;
  }

  if (total_variation > kMaxTotalVariation) {
    std::cerr << "[VALIDATION][MPI] total variation too large (" << total_variation << ")\n";
    return false;
  }

  return true;
}

bool SizovDGlobalSearchMPI::PreProcessingImpl() {
  const auto &problem = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    std::cout << "[DEBUG][ops_mpi.cpp] START: left=" << problem.left << " right=" << problem.right
              << " accuracy=" << problem.accuracy << " reliability=" << problem.reliability
              << " max_iter=" << problem.max_iterations << "\n";
  }

  std::vector<Seed> seeds;
  if (!CollectInitialSeeds(problem, seeds)) {
    if (rank == 0) {
      std::cerr << "[DEBUG][ops_mpi.cpp] Unable to sample function on interval\n";
    }
    return false;
  }

  std::sort(seeds.begin(), seeds.end(), [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
  const double uniq_eps = std::max(problem.accuracy * 0.5, kMinInsertEps);

  std::vector<Seed> unique_seeds;
  unique_seeds.reserve(seeds.size());
  for (const auto &seed : seeds) {
    if (!unique_seeds.empty() && std::abs(seed.first - unique_seeds.back().first) <= uniq_eps) {
      continue;
    }
    unique_seeds.push_back(seed);
  }

  ExpandSeedsAroundBest(problem, uniq_eps, unique_seeds);

  if (unique_seeds.size() < 2) {
    if (rank == 0) {
      std::cerr << "[DEBUG][ops_mpi.cpp] Not enough unique samples after filtering\n";
    }
    return false;
  }

  InitializeStateFromSeeds(unique_seeds);

  if (rank == 0) {
    std::cout << "[DEBUG][ops_mpi.cpp] Initial points count: " << points_.size() << "\n";
    std::cout << "[DEBUG][ops_mpi.cpp] Initial best: x=" << best_point_ << " f=" << best_value_ << "\n";
  }

  SaveOutput();
  return true;
}

bool SizovDGlobalSearchMPI::RunImpl() {
  const auto &problem = GetInput();

  if (!HasEnoughPoints()) {
    return false;
  }

  const double span_total = problem.right - problem.left;
  const double insert_eps = ComputeInsertEps(problem.accuracy, span_total);

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int performed_iterations = 0;

  for (int iter = 0; iter < problem.max_iterations; ++iter) {
    performed_iterations = iter + 1;
    const double m = EstimateM(problem.reliability);

    double best_characteristic = -std::numeric_limits<double>::infinity();
    const auto best_interval = SelectIntervalByCharacteristicMPI(m, &best_characteristic, rank, size);

    if (rank == 0 && (iter % 50 == 0)) {
      if (best_interval) {
        const std::size_t idx = *best_interval;
        std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] M=" << m << " points=" << points_.size()
                  << " best_interval=(" << points_[idx - 1] << ", " << points_[idx] << ") characteristic="
                  << best_characteristic << "\n";
      } else {
        std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] STOP: no valid intervals\n";
      }
    }

    if (!best_interval) {
      converged_ = true;
      break;
    }

    const std::size_t best_right_idx = *best_interval;
    const double interval = points_[best_right_idx] - points_[best_right_idx - 1];
    if (ShouldStopOnInterval(interval, problem.accuracy)) {
      if (rank == 0 && (iter % 50 == 0)) {
        std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] STOP: interval length " << interval << " <= accuracy "
                  << problem.accuracy << "\n";
      }
      converged_ = true;
      break;
    }

    ProcessBestInterval(problem, best_right_idx, m, insert_eps, span_total, rank, iter);

    if (ShouldPerformLocalScan(iter)) {
      ScanAroundBestMinimum(problem, span_total, insert_eps);
    }
  }

  iterations_ = performed_iterations;

  const bool refined = RefineAroundBest(problem);
  if (refined && !converged_) {
    converged_ = true;
  }

  if (rank == 0) {
    std::cout << "[DEBUG][ops_mpi.cpp] FINISH: converged=" << converged_ << " best_x=" << best_point_
              << " best_f=" << best_value_ << " iterations=" << iterations_ << "\n";
  }

  SaveOutput();
  return true;
}

bool SizovDGlobalSearchMPI::PostProcessingImpl() { return true; }

bool SizovDGlobalSearchMPI::CollectInitialSeeds(const InType &problem, std::vector<Seed> &seeds) const {
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

  const double mid_point = Midpoint(problem.left, problem.right);
  if (auto value = EvaluatePoint(problem, mid_point, false)) {
    seeds.emplace_back(mid_point, *value);
  }

  constexpr int kExtraSamples = 10;
  for (int i = 1; i < kExtraSamples; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(kExtraSamples);
    const double point = problem.left + t * (problem.right - problem.left);
    if (auto value = EvaluatePoint(problem, point, false)) {
      seeds.emplace_back(point, *value);
    }
  }

  return seeds.size() >= 2;
}

void SizovDGlobalSearchMPI::ExpandSeedsAroundBest(const InType &problem, double uniq_eps,
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
  const double focus_radius = std::max(problem.accuracy * 600.0, span * 0.02);
  if (focus_radius <= problem.accuracy) {
    return;
  }

  constexpr int kFocusSamples = 12;
  for (int i = 0; i < kFocusSamples; ++i) {
    const double frac = (static_cast<double>(i) / static_cast<double>(kFocusSamples - 1)) * 2.0 - 1.0;
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

void SizovDGlobalSearchMPI::InitializeStateFromSeeds(const std::vector<Seed> &seeds) {
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

std::optional<double> SizovDGlobalSearchMPI::EvaluatePoint(const InType &problem, double point,
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

bool SizovDGlobalSearchMPI::SeekFiniteSample(const InType &problem, double start, double direction, double &point,
                                             double &value) const {
  const double span = problem.right - problem.left;
  double step = std::max(problem.accuracy, span * 1e-4);
  point = start;
  for (int attempt = 0; attempt < kMaxSeekAttempts; ++attempt) {
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

double SizovDGlobalSearchMPI::DynamicLimit(double sample) const {
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

double SizovDGlobalSearchMPI::ClampValue(double value, bool *clipped) const {
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

double SizovDGlobalSearchMPI::EstimateM(double reliability) const {
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
  return std::clamp(m, kMinM, kMaxM);
}

std::pair<double, bool> SizovDGlobalSearchMPI::EvaluateIntervalCandidate(const InType &problem, double candidate,
                                                                         double left_point, double right_point,
                                                                         double left_value, double right_value) const {
  constexpr double kMaxSlope = 1e3;

  if (auto value = EvaluatePoint(problem, candidate, true)) {
    if (SlopesAreAcceptable(*value, left_value, right_value, candidate, left_point, right_point, problem.accuracy,
                            kMaxSlope)) {
      return {*value, false};
    }
  }

  const double span = right_point - left_point;
  double step = std::max(problem.accuracy, span * 1e-4);
  for (int attempt = 0; attempt < kMaxShiftAttempts; ++attempt) {
    const double offset = ((attempt / 2) + 1) * step;
    double shifted = candidate + ((attempt % 2 == 0) ? offset : -offset);

    if (shifted <= left_point) {
      shifted = std::nextafter(left_point, right_point);
    }
    if (shifted >= right_point) {
      shifted = std::nextafter(right_point, left_point);
    }

    if (auto value = EvaluatePoint(problem, shifted, true)) {
      if (SlopesAreAcceptable(*value, left_value, right_value, shifted, left_point, right_point, problem.accuracy,
                              kMaxSlope)) {
        return {*value, false};
      }
    }
    step *= 1.5;
  }

  double fallback = ComputeFallbackValue(left_value, right_value);
  bool clipped = false;
  double clamped = ClampValue(fallback, &clipped);
  return {clamped, true};
}

double SizovDGlobalSearchMPI::FilterCandidateValue(double candidate_value, double left_value, double right_value,
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

bool SizovDGlobalSearchMPI::TryInsertPoint(double point, double value, double insert_eps) {
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

std::optional<std::size_t> SizovDGlobalSearchMPI::SelectIntervalByCharacteristicMPI(double m,
                                                                                   double *best_characteristic,
                                                                                   int rank, int size) const {
  if (points_.size() < 2) {
    return std::nullopt;
  }
  const std::size_t interval_count = points_.size() - 1;

  auto [chunk_begin, chunk_end] = GetChunk(interval_count, size, rank);
  IntervalCharacteristic local;

  for (std::size_t interval = chunk_begin; interval < chunk_end; ++interval) {
    const std::size_t right_idx = interval + 1;
    if (clipped_[right_idx] || clipped_[right_idx - 1]) {
      continue;
    }
    const double characteristic = detail_mpi::CalcCharacteristic(points_, values_, right_idx, m);
    if (characteristic > local.characteristic) {
      local.characteristic = characteristic;
      local.right_index = static_cast<int>(right_idx);
    }
  }

  IntervalCharacteristic global;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  if (global.right_index <= 0 || static_cast<std::size_t>(global.right_index) >= points_.size() ||
      !std::isfinite(global.characteristic)) {
    return std::nullopt;
  }

  if (best_characteristic) {
    *best_characteristic = global.characteristic;
  }
  return static_cast<std::size_t>(global.right_index);
}

void SizovDGlobalSearchMPI::ScanAroundBestMinimum(const InType &problem, double span_total, double insert_eps) {
  const double radius = std::max(problem.accuracy * 500.0, span_total * 0.006);
  const double scan_left = std::max(problem.left, best_point_ - radius);
  const double scan_right = std::min(problem.right, best_point_ + radius);
  if (scan_right <= scan_left + problem.accuracy) {
    return;
  }

  for (int i = 0; i <= kLocalScanSamples; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(kLocalScanSamples);
    const double point = scan_left + t * (scan_right - scan_left);
    if (auto value = EvaluatePoint(problem, point, true)) {
      TryInsertPoint(point, *value, insert_eps);
    }
  }
}

bool SizovDGlobalSearchMPI::RefineAroundBest(const InType &problem) {
  bool refined = false;
  const double refine_radius = std::max(kMinRefineRadius, problem.accuracy * 500.0);
  const double refine_left = std::max(problem.left, best_point_ - refine_radius);
  const double refine_right = std::min(problem.right, best_point_ + refine_radius);
  if (refine_right <= refine_left) {
    return false;
  }

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
  return refined;
}

bool SizovDGlobalSearchMPI::HasEnoughPoints() const { return points_.size() >= 2; }

double SizovDGlobalSearchMPI::ComputeInsertEps(double accuracy, double span_total) {
  const double span_term = std::max(span_total * 1e-9, 0.0);
  const double accuracy_term = std::max(accuracy * 0.5, kMinInsertEps);
  return std::max(span_term, accuracy_term);
}

bool SizovDGlobalSearchMPI::ShouldStopOnInterval(double interval, double accuracy) { return interval <= accuracy; }

bool SizovDGlobalSearchMPI::ShouldPerformLocalScan(int iter) const {
  return best_value_ < 0.0 && ((iter + 1) % kLocalScanPeriod == 0);
}

void SizovDGlobalSearchMPI::ProcessBestInterval(const InType &problem, std::size_t best_right_idx, double m,
                                                double insert_eps, double span_total, int rank, int iter) {
  const double left_point = points_[best_right_idx - 1];
  const double right_point = points_[best_right_idx];
  const double interval = right_point - left_point;

  const double interval_fraction = interval / std::max(span_total, std::numeric_limits<double>::epsilon());
  const double local_min_m = std::max(0.5, interval_fraction * 100.0);
  const double local_max_m = std::max(150.0, interval_fraction * 3000.0);
  const double m_interval = std::clamp(m, local_min_m, local_max_m);

  double new_point = detail_mpi::CalcNewPoint(points_, values_, best_right_idx, m_interval);
  if (!(new_point > left_point && new_point < right_point)) {
    new_point = Midpoint(left_point, right_point);
  }

  const double left_value = values_[best_right_idx - 1];
  const double right_value = values_[best_right_idx];
  auto eval_res =
      EvaluateIntervalCandidate(problem, new_point, left_point, right_point, left_value, right_value);
  bool clipped = eval_res.second;
  double new_value = FilterCandidateValue(eval_res.first, left_value, right_value, &clipped);

  TryInsertPoint(new_point, new_value, insert_eps);

  if (rank == 0 && (iter % 50 == 0)) {
    std::cout << "[DEBUG][ops_mpi.cpp][iter=" << iter << "] New point: x=" << new_point << " f(x)=" << new_value
              << "\n";
  }
}

void SizovDGlobalSearchMPI::SaveOutput() {
  GetOutput() = Solution{
      .argmin = best_point_,
      .value = best_value_,
      .iterations = iterations_,
      .converged = converged_,
  };
}

}  // namespace sizov_d_global_search
