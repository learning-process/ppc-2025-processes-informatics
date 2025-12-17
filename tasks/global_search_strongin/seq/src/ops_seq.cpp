#include "global_search_strongin/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "global_search_strongin/common/include/common.hpp"

namespace global_search_strongin {
namespace {

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
    const auto interval_index = SelectInterval();
    if (!interval_index.has_value()) {
      break;
    }
    if (!InsertMidpoint(input, *interval_index, input.epsilon)) {
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

std::optional<std::size_t> StronginSearchSeq::SelectInterval() const {
  if (points_.size() < 2) {
    return std::nullopt;
  }
  double best_interval_score = std::numeric_limits<double>::infinity();
  double best_interval_length = 0.0;
  std::size_t best_interval_index = 1;

  for (std::size_t i = 1; i < points_.size(); ++i) {
    const double length = points_[i].x - points_[i - 1].x;
    if (length <= 0.0) {
      continue;
    }
    const double score = std::min(points_[i - 1].value, points_[i].value);
    const bool better_score = score < best_interval_score;
    const bool longer_equal_score = std::fabs(score - best_interval_score) < 1e-12 && length > best_interval_length;
    if (better_score || longer_equal_score) {
      best_interval_score = score;
      best_interval_length = length;
      best_interval_index = i;
    }
  }

  if (!std::isfinite(best_interval_score)) {
    return std::nullopt;
  }
  return best_interval_index;
}

bool StronginSearchSeq::InsertMidpoint(const InType &input, std::size_t interval_index, double epsilon) {
  const SamplePoint &left = points_[interval_index - 1];
  const SamplePoint &right = points_[interval_index];
  const double interval_length = right.x - left.x;
  if (interval_length < epsilon) {
    return false;
  }

  const double midpoint = 0.5 * (left.x + right.x);
  if (midpoint <= input.left || midpoint >= input.right) {
    return false;
  }

  const bool already_used = std::ranges::any_of(points_, [midpoint](const SamplePoint &point) {
    return std::fabs(point.x - midpoint) < std::numeric_limits<double>::epsilon();
  });
  if (already_used) {
    return false;
  }

  const double value = Evaluate(input, midpoint);
  if (value < best_value_) {
    best_value_ = value;
    best_x_ = midpoint;
  }

  SamplePoint new_point{.x = midpoint, .value = value};
  auto insert_it = std::ranges::upper_bound(points_, new_point, Comparator);
  points_.insert(insert_it, new_point);
  return true;
}

}  // namespace global_search_strongin
