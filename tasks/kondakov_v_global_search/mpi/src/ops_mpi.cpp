#include "kondakov_v_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "kondakov_v_global_search/common/include/common.hpp"

namespace kondakov_v_global_search {

KondakovVGlobalSearchMPI::KondakovVGlobalSearchMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

double KondakovVGlobalSearchMPI::EvaluateFunction(double x) {
  const auto &cfg = GetInput();
  switch (cfg.func_type) {
    case FunctionType::kQuadratic: {
      double t = cfg.func_param;
      return (x - t) * (x - t);
    }
    case FunctionType::kSine:
      return std::sin(x) + (0.1 * x);
    case FunctionType::kAbs:
      return std::abs(x);
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

bool KondakovVGlobalSearchMPI::IsRoot() const {
  return world_rank_ == 0;
}

bool KondakovVGlobalSearchMPI::ValidationImpl() {
  const auto &cfg = GetInput();
  bool local_valid = cfg.left < cfg.right && cfg.accuracy > 0.0 && cfg.reliability > 0.0 && cfg.max_iterations > 0;

  bool global_valid = false;
  MPI_Allreduce(&local_valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  return global_valid;
}

bool KondakovVGlobalSearchMPI::PreProcessingImpl() {
  const auto &cfg = GetInput();

  if (IsRoot()) {
    points_x_.clear();
    values_y_.clear();
    points_x_.reserve(cfg.max_iterations + (world_size_ * 10));
    values_y_.reserve(cfg.max_iterations + (world_size_ * 10));

    double f_a = EvaluateFunction(cfg.left);
    double f_b = EvaluateFunction(cfg.right);
    if (!std::isfinite(f_a) || !std::isfinite(f_b)) {
      return false;
    }

    points_x_ = {cfg.left, cfg.right};
    values_y_ = {f_a, f_b};
    best_point_ = (f_a < f_b) ? cfg.left : cfg.right;
    best_value_ = std::min(f_a, f_b);
  }

  SyncGlobalState();
  return true;
}

void KondakovVGlobalSearchMPI::SyncGlobalState() {
  int n = IsRoot() ? static_cast<int>(points_x_.size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (!IsRoot()) {
    points_x_.resize(n);
    values_y_.resize(n);
  }

  if (n > 0) {
    MPI_Bcast(points_x_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(values_y_.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  best_value_ = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < points_x_.size(); ++i) {
    if (values_y_[i] < best_value_) {
      best_value_ = values_y_[i];
      best_point_ = points_x_[i];
    }
  }
}

double KondakovVGlobalSearchMPI::ComputeAdaptiveLipschitzEstimate(double r) const {
  const double min_slope = 1e-2;
  if (points_x_.size() < 2) {
    return r * min_slope;
  }

  double max_slope = min_slope;
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
    if (std::isfinite(slope) && slope > max_slope) {
      max_slope = slope;
    }
  }
  return r * max_slope;
}

double KondakovVGlobalSearchMPI::IntervalMerit(std::size_t i, double l_est) const {
  double x_l = points_x_[i - 1];
  double x_r = points_x_[i];
  double f_l = values_y_[i - 1];
  double f_r = values_y_[i];
  double h = x_r - x_l;
  double df = f_r - f_l;
  return (l_est * h) - (2.0 * (f_l + f_r)) + ((df * df) / (l_est * h));
}

double KondakovVGlobalSearchMPI::ProposeTrialPoint(std::size_t i, double l_est) const {
  double x_l = points_x_[i - 1];
  double x_r = points_x_[i];
  double f_l = values_y_[i - 1];
  double f_r = values_y_[i];
  double mid = 0.5 * (x_l + x_r);
  double asym = (f_r - f_l) / (2.0 * l_est);
  double cand = mid - asym;
  if (cand <= x_l || cand >= x_r) {
    cand = mid;
  }
  return cand;
}

std::size_t KondakovVGlobalSearchMPI::LocateInsertionIndex(double x) const {
  auto it = std::ranges::lower_bound(points_x_, x);
  return static_cast<std::size_t>(std::distance(points_x_.begin(), it));
}

void KondakovVGlobalSearchMPI::InsertEvaluation(double x, double fx) {
  auto idx = LocateInsertionIndex(x);
  points_x_.insert(points_x_.begin() + static_cast<std::vector<double>::difference_type>(idx), x);
  values_y_.insert(values_y_.begin() + static_cast<std::vector<double>::difference_type>(idx), fx);
  if (fx < best_value_) {
    best_value_ = fx;
    best_point_ = x;
  }
}

void KondakovVGlobalSearchMPI::SelectIntervalsToRefine(double l_est,
                                                       std::vector<std::pair<double, std::size_t>> &merits) {
  merits.clear();
  for (std::size_t i = 1; i < points_x_.size(); ++i) {
    merits.emplace_back(IntervalMerit(i, l_est), i);
  }
  std::ranges::sort(merits, [](const auto &a, const auto &b) { return a.first > b.first; });
}

bool KondakovVGlobalSearchMPI::CheckConvergence(const Params &cfg,
                                                const std::vector<std::pair<double, std::size_t>> &merits) {
  if (merits.empty()) {
    return false;
  }
  double width = points_x_[merits[0].second] - points_x_[merits[0].second - 1];
  if (width <= cfg.accuracy) {
    has_converged_ = true;
    return true;
  }
  return false;
}

void KondakovVGlobalSearchMPI::GatherAndBroadcastTrialResults(const std::vector<std::pair<double, std::size_t>> &merits,
                                                              int num_trials, double l_est) {
  double local_x = 0.0;
  double local_fx = 0.0;
  int local_count = 0;

  if (world_rank_ < num_trials && !merits.empty()) {
    std::size_t idx = merits[world_rank_].second;
    double x = ProposeTrialPoint(idx, l_est);
    double fx = EvaluateFunction(x);
    if (std::isfinite(fx)) {
      local_x = x;
      local_fx = fx;
      local_count = 2;
    }
  }

  std::vector<int> counts(world_size_);
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::vector<int> displs(world_size_);
  for (int i = 1; i < world_size_; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }
  int total = (world_size_ > 0) ? (displs.back() + counts.back()) : 0;

  std::vector<double> recv_buf(total);
  std::array<double, 2> send_buf = {local_x, local_fx};
  const double *send_ptr = (local_count > 0) ? send_buf.data() : nullptr;

  MPI_Allgatherv(send_ptr, local_count, MPI_DOUBLE, recv_buf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  if (IsRoot()) {
    for (int i = 0; i < total; i += 2) {
      InsertEvaluation(recv_buf[i], recv_buf[i + 1]);
    }
  }
}

bool KondakovVGlobalSearchMPI::RunImpl() {
  const auto &cfg = GetInput();
  std::vector<std::pair<double, std::size_t>> merits;
  double l_est = ComputeAdaptiveLipschitzEstimate(cfg.reliability);

  for (int step = 0; step < cfg.max_iterations; ++step) {
    if (step % 10 == 0) {
      l_est = ComputeAdaptiveLipschitzEstimate(cfg.reliability);
    }

    SelectIntervalsToRefine(l_est, merits);
    if (CheckConvergence(cfg, merits)) {
      break;
    }

    int num_trials = std::min(static_cast<int>(merits.size()), world_size_);
    GatherAndBroadcastTrialResults(merits, num_trials, l_est);
    SyncGlobalState();
    total_evals_ += num_trials;
  }

  if (IsRoot()) {
    GetOutput() =
        Solution{.argmin = best_point_, .value = best_value_, .iterations = total_evals_, .converged = has_converged_};
  }
  return true;
}

bool KondakovVGlobalSearchMPI::PostProcessingImpl() {
  Solution sol;
  if (IsRoot()) {
    sol = GetOutput();
  }

  MPI_Bcast(&sol.argmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sol.value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sol.iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int converged = sol.converged ? 1 : 0;
  MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
  sol.converged = (converged != 0);

  GetOutput() = sol;
  return true;
}

}  // namespace kondakov_v_global_search
