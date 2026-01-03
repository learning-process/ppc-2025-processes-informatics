#include "global_search_strongin/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

#include "global_search_strongin/common/include/common.hpp"

namespace global_search_strongin {
namespace {

constexpr double kNearEps = 1e-12;

double Evaluate(const InType &input, double x) {
  return input.objective ? input.objective(x) : 0.0;
}

bool Comparator(const SamplePoint &lhs, const SamplePoint &rhs) {
  return lhs.x < rhs.x;
}

}  // namespace

StronginSearchSeq::StronginSearchSeq(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool StronginSearchSeq::ValidationImpl() {
  const auto &input = GetInput();
  if (!input.objective) {
    return false;
  }
  if (!(input.left < input.right)) {
    return false;
  }
  if (input.epsilon <= 0.0 || input.reliability <= 0.0) {
    return false;
  }
  return input.max_iterations > 0;
}

bool StronginSearchSeq::PreProcessingImpl() {
  points_.clear();
  const auto &input = GetInput();

  const SamplePoint left{.x = input.left, .value = Evaluate(input, input.left)};
  const SamplePoint right{.x = input.right, .value = Evaluate(input, input.right)};
  points_.push_back(left);
  points_.push_back(right);
  if (!std::ranges::is_sorted(points_, Comparator)) {
    std::ranges::sort(points_, Comparator);
  }

  best_x_ = left.value < right.value ? left.x : right.x;
  best_value_ = std::min(left.value, right.value);
  iterations_done_ = 0;
  return true;
}

bool StronginSearchSeq::RunImpl() {
  const auto &input = GetInput();

  while (iterations_done_ < input.max_iterations) {
    if (points_.size() < 2) {
      break;
    }

    const double max_slope = ComputeMaxSlope();
    const double m = max_slope > 0.0 ? input.reliability * max_slope : 1.0;
    const auto interval_index = SelectInterval(m);
    if (!interval_index.has_value()) {
      break;
    }

    if (!InsertPoint(input, *interval_index, input.epsilon, m)) {
      break;
    }

    ++iterations_done_;
  }

  return true;
}

bool StronginSearchSeq::PostProcessingImpl() {
  OutType out{};
  out.best_point = best_x_;
  out.best_value = best_value_;
  out.iterations = iterations_done_;
  GetOutput() = out;
  return true;
}

double StronginSearchSeq::ComputeMaxSlope() const {
  if (points_.size() < 2) {
    return 0.0;
  }
  double max_slope = 0.0;
  for (std::size_t i = 1; i < points_.size(); ++i) {
    const auto &left = points_[i - 1];
    const auto &right = points_[i];
    const double delta = right.x - left.x;
    if (delta <= 0.0) {
      continue;
    }
    const double slope = std::fabs(right.value - left.value) / delta;
    max_slope = std::max(max_slope, slope);
  }
  return max_slope;
}

std::optional<std::size_t> StronginSearchSeq::SelectInterval(double m) const {
  if (points_.size() < 2) {
    return std::nullopt;
  }

  double best_characteristic = -std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index = std::nullopt;

  for (std::size_t i = 1; i < points_.size(); ++i) {
    const auto &left = points_[i - 1];
    const auto &right = points_[i];
    const double delta = right.x - left.x;
    if (delta <= 0.0) {
      continue;
    }

    const double diff = right.value - left.value;
    const double characteristic = (m * delta) + ((diff * diff) / (m * delta)) - (2.0 * (right.value + left.value));

    if (characteristic > best_characteristic) {
      best_characteristic = characteristic;
      best_index = i;
    }
  }

  if (!std::isfinite(best_characteristic)) {
    return std::nullopt;
  }
  return best_index;
}

bool StronginSearchSeq::InsertPoint(const InType &input, std::size_t interval_index, double epsilon, double m) {
  if (interval_index == 0 || interval_index >= points_.size()) {
    return false;
  }

  const SamplePoint &left = points_[interval_index - 1];
  const SamplePoint &right = points_[interval_index];
  const double interval_length = right.x - left.x;
  if (interval_length < epsilon) {
    return false;
  }

  double x_new = (0.5 * (left.x + right.x)) - ((right.value - left.value) / (2.0 * m));
  x_new = std::clamp(x_new, input.left, input.right);

  if (x_new <= left.x + kNearEps || x_new >= right.x - kNearEps) {
    x_new = 0.5 * (left.x + right.x);
  }

  if (x_new <= input.left || x_new >= input.right) {
    return false;
  }

  const bool already_used =
      std::ranges::any_of(points_, [x_new](const SamplePoint &p) { return std::fabs(p.x - x_new) < kNearEps; });
  if (already_used) {
    return false;
  }

  const double value = Evaluate(input, x_new);
  if (value < best_value_) {
    best_value_ = value;
    best_x_ = x_new;
  }

  SamplePoint new_point{.x = x_new, .value = value};
  auto insert_it = std::ranges::upper_bound(points_, new_point, Comparator);
  points_.insert(insert_it, new_point);
  return true;
}

}  // namespace global_search_strongin
