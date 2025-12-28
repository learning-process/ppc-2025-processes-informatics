#include "kondakov_v_global_search/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
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

bool KondakovVGlobalSearchMPI::IsRoot() const {
  return world_rank_ == 0;
}

bool KondakovVGlobalSearchMPI::ValidationImpl() {
  const auto &cfg = GetInput();
  bool local_valid =
      cfg.func && cfg.left < cfg.right && cfg.accuracy > 0.0 && cfg.reliability > 0.0 && cfg.max_iterations > 0;

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

    double f_a = cfg.func(cfg.left);
    double f_b = cfg.func(cfg.right);
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
    points_x_.resize(static_cast<std::size_t>(n));
    values_y_.resize(static_cast<std::size_t>(n));
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
  return std::ranges::lower_bound(points_x_, x) - points_x_.begin();
}

void KondakovVGlobalSearchMPI::InsertEvaluation(double x, double fx) {
  auto idx = LocateInsertionIndex(x);
  auto pos = static_cast<std::vector<double>::difference_type>(idx);
  points_x_.insert(points_x_.begin() + pos, x);
  values_y_.insert(values_y_.begin() + pos, fx);
  if (fx < best_value_) {
    best_value_ = fx;
    best_point_ = x;
  }
}

void KondakovVGlobalSearchMPI::SelectIntervalsToRefine(double l_est,
                                                       std::vector<std::pair<double, std::size_t>> &merits) {
  merits.clear();
  for (std::size_t i = 1; i < points_x_.size(); ++i) {
    double m = IntervalMerit(i, l_est);
    merits.emplace_back(m, i);
  }
  std::ranges::sort(merits, [](const auto &a, const auto &b) { return a.first > b.first; });
}

bool KondakovVGlobalSearchMPI::CheckConvergence(const Params &cfg,
                                                const std::vector<std::pair<double, std::size_t>> &merits) {
  if (merits.empty()) {
    return false;
  }
  std::size_t best_i = merits[0].second;
  double width = points_x_[best_i] - points_x_[best_i - 1];
  if (width <= cfg.accuracy) {
    has_converged_ = true;
    return true;
  }
  return false;
}

void KondakovVGlobalSearchMPI::GatherAndBroadcastTrialResults(const std::vector<std::pair<double, std::size_t>> &merits,
                                                              int num_trials, double l_est, const Params &cfg) {
  double local_x = 0.0;
  double local_fx = 0.0;
  bool has_local_trial = false;

  if (world_rank_ < num_trials) {
    std::size_t interval_idx = merits[world_rank_].second;
    local_x = ProposeTrialPoint(interval_idx, l_est);
    local_fx = cfg.func(local_x);
    has_local_trial = std::isfinite(local_fx);
  }

  std::vector<int> send_counts(world_size_);
  std::vector<int> send_displs(world_size_);
  std::vector<double> local_pair = has_local_trial ? std::vector<double>{local_x, local_fx} : std::vector<double>{};

  int local_count = static_cast<int>(local_pair.size());
  MPI_Allgather(&local_count, 1, MPI_INT, send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  send_displs[0] = 0;
  for (int i = 1; i < world_size_; ++i) {
    send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
  }
  int total_recv = send_displs.back() + send_counts.back();

  std::vector<double> recv_buf(total_recv);
  MPI_Allgatherv(local_pair.data(), local_count, MPI_DOUBLE, recv_buf.data(), send_counts.data(), send_displs.data(),
                 MPI_DOUBLE, MPI_COMM_WORLD);

  if (IsRoot()) {
    for (int i = 0; i < total_recv; i += 2) {
      double x = recv_buf[i];
      double fx = recv_buf[i + 1];
      InsertEvaluation(x, fx);
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
    GatherAndBroadcastTrialResults(merits, num_trials, l_est, cfg);

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
  Solution sol{};
  if (IsRoot()) {
    sol = GetOutput();
  }

  MPI_Bcast(&sol.argmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sol.value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&sol.iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int converged_int = sol.converged ? 1 : 0;
  MPI_Bcast(&converged_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  sol.converged = (converged_int != 0);

  GetOutput() = sol;
  return true;
}

}  // namespace kondakov_v_global_search
