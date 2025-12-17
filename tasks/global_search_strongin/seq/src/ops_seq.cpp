#include "global_search_strongin/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace global_search_strongin {
namespace {

double Evaluate(const InType &input, double x) {
  return input.objective ? input.objective(x) : 0.0;
}

bool PointsAreSorted(const std::vector<SamplePoint> &points) {
  for (std::size_t i = 1; i < points.size(); ++i) {
    if (points[i - 1].x >= points[i].x) {
      return false;
    }
  }
  return true;
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
  const SamplePoint left{input.left, Evaluate(input, input.left)};
  const SamplePoint right{input.right, Evaluate(input, input.right)};
  points_.push_back(left);
  points_.push_back(right);
  if (!PointsAreSorted(points_)) {
    std::sort(points_.begin(), points_.end(),
              [](const SamplePoint &lhs, const SamplePoint &rhs) { return lhs.x < rhs.x; });
  }
  best_x_ = left.value < right.value ? left.x : right.x;
  best_value_ = std::min(left.value, right.value);
  iterations_done_ = 0;
  return true;
}

bool StronginSearchSeq::RunImpl() {
  const auto &input = GetInput();
  const double epsilon = input.epsilon;
  const int max_iterations = input.max_iterations;

  while (iterations_done_ < max_iterations) {
    double best_interval_score = std::numeric_limits<double>::infinity();
    double best_interval_length = 0.0;
    std::size_t best_interval_index = 1;

    for (std::size_t i = 1; i < points_.size(); ++i) {
      const double length = points_[i].x - points_[i - 1].x;
      if (length <= 0.0) {
        continue;
      }
      const double score = std::min(points_[i - 1].value, points_[i].value);
      if (score < best_interval_score ||
          (std::fabs(score - best_interval_score) < 1e-12 && length > best_interval_length)) {
        best_interval_score = score;
        best_interval_length = length;
        best_interval_index = i;
      }
    }

    const SamplePoint &left = points_[best_interval_index - 1];
    const SamplePoint &right = points_[best_interval_index];
    if ((right.x - left.x) < epsilon) {
      break;
    }

    const double midpoint = 0.5 * (left.x + right.x);
    if (midpoint <= input.left || midpoint >= input.right) {
      break;
    }

    bool already_used = std::any_of(points_.begin(), points_.end(), [midpoint](const SamplePoint &point) {
      return std::fabs(point.x - midpoint) < std::numeric_limits<double>::epsilon();
    });
    if (already_used) {
      break;
    }

    const double value = Evaluate(input, midpoint);
    if (value < best_value_) {
      best_value_ = value;
      best_x_ = midpoint;
    }

    SamplePoint new_point{midpoint, value};
    points_.insert(std::upper_bound(points_.begin(), points_.end(), new_point,
                                    [](const SamplePoint &lhs, const SamplePoint &rhs) { return lhs.x < rhs.x; }),
                   new_point);
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

}  // namespace global_search_strongin
