#include "global_search_strongin/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace global_search_strongin {
namespace {

double Evaluate(const InType &input, double x) {
  return input.objective ? input.objective(x) : 0.0;
}

bool Comparator(const SamplePoint &lhs, const SamplePoint &rhs) {
  return lhs.x < rhs.x;
}

}  // namespace

StronginSearchMpi::StronginSearchMpi(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool StronginSearchMpi::ValidationImpl() {
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

bool StronginSearchMpi::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  points_.clear();
  const auto &input = GetInput();
  const SamplePoint left{input.left, Evaluate(input, input.left)};
  const SamplePoint right{input.right, Evaluate(input, input.right)};
  points_.push_back(left);
  points_.push_back(right);
  if (!std::is_sorted(points_.begin(), points_.end(), Comparator)) {
    std::sort(points_.begin(), points_.end(), Comparator);
  }
  best_x_ = left.value < right.value ? left.x : right.x;
  best_value_ = std::min(left.value, right.value);
  iterations_done_ = 0;
  return true;
}

bool StronginSearchMpi::RunImpl() {
  const auto &input = GetInput();
  const double reliability = input.reliability;
  const double epsilon = input.epsilon;
  const int max_iterations = input.max_iterations;

  while (iterations_done_ < max_iterations) {
    if (points_.size() < 2) {
      break;
    }
    const int intervals = static_cast<int>(points_.size()) - 1;

    double local_max_slope = 0.0;
    for (int i = 0; i < intervals; ++i) {
      const double delta = points_[static_cast<std::size_t>(i + 1)].x - points_[static_cast<std::size_t>(i)].x;
      if (delta <= 0.0) {
        continue;
      }
      const double diff = points_[static_cast<std::size_t>(i + 1)].value - points_[static_cast<std::size_t>(i)].value;
      const double slope = std::fabs(diff) / delta;
      local_max_slope = std::max(local_max_slope, slope);
    }

    double global_max_slope = 0.0;
    MPI_Allreduce(&local_max_slope, &global_max_slope, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double m = global_max_slope > 0.0 ? reliability * global_max_slope : 1.0;

    const int base = intervals / world_size_;
    const int remainder = intervals % world_size_;
    const int start = base * rank_ + std::min(rank_, remainder);
    const int count = base + (rank_ < remainder ? 1 : 0);
    const int end = start + count;

    double local_best_value = -std::numeric_limits<double>::infinity();
    int local_best_index = -1;

    for (int idx = start; idx < end; ++idx) {
      const auto &left = points_[static_cast<std::size_t>(idx)];
      const auto &right = points_[static_cast<std::size_t>(idx + 1)];
      const double delta = right.x - left.x;
      if (delta <= 0.0) {
        continue;
      }
      const double diff = right.value - left.value;
      const double candidate = m * delta + (diff * diff) / (m * delta) - 2.0 * (right.value + left.value);
      if (candidate > local_best_value) {
        local_best_value = candidate;
        local_best_index = idx;
      }
    }

    struct {
      double value;
      int index;
    } local_pair{local_best_value, local_best_index}, global_pair{};

    MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    const int best_index = global_pair.index;
    if (best_index < 0 || best_index >= intervals) {
      break;
    }

    const auto &left = points_[static_cast<std::size_t>(best_index)];
    const auto &right = points_[static_cast<std::size_t>(best_index + 1)];
    const double interval_length = right.x - left.x;

    int continue_flag = 1;
    double new_point = 0.0;
    double new_value = 0.0;
    int insert_index = 0;

    if (rank_ == 0) {
      if (interval_length < epsilon) {
        continue_flag = 0;
      } else {
        new_point = 0.5 * (left.x + right.x) - (right.value - left.value) / (2.0 * m);
        new_point = std::clamp(new_point, input.left, input.right);
        const bool already_used = std::any_of(points_.begin(), points_.end(), [new_point](const SamplePoint &point) {
          return std::fabs(point.x - new_point) < std::numeric_limits<double>::epsilon();
        });
        if (already_used) {
          continue_flag = 0;
        } else {
          new_value = Evaluate(input, new_point);
          if (new_value < best_value_) {
            best_value_ = new_value;
            best_x_ = new_point;
          }
          auto insert_it =
              std::upper_bound(points_.begin(), points_.end(), SamplePoint{new_point, new_value}, Comparator);
          insert_index = static_cast<int>(std::distance(points_.begin(), insert_it));
          points_.insert(insert_it, SamplePoint{new_point, new_value});
        }
      }
    }

    MPI_Bcast(&continue_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (continue_flag == 0) {
      break;
    }

    MPI_Bcast(&insert_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&new_point, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&new_value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank_ != 0) {
      points_.insert(points_.begin() + insert_index, SamplePoint{new_point, new_value});
      if (new_value < best_value_) {
        best_value_ = new_value;
        best_x_ = new_point;
      }
    }

    double best_data[2] = {best_x_, best_value_};
    MPI_Bcast(best_data, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    best_x_ = best_data[0];
    best_value_ = best_data[1];

    ++iterations_done_;
  }

  return true;
}

bool StronginSearchMpi::PostProcessingImpl() {
  OutType out{};
  if (rank_ == 0) {
    out.best_point = best_x_;
    out.best_value = best_value_;
    out.iterations = iterations_done_;
    GetOutput() = out;
  }

  double point_value_data[2] = {best_x_, best_value_};
  MPI_Bcast(point_value_data, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int iterations = iterations_done_;
  MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    auto &output = GetOutput();
    output.best_point = point_value_data[0];
    output.best_value = point_value_data[1];
    output.iterations = iterations;
  }
  return true;
}

}  // namespace global_search_strongin
