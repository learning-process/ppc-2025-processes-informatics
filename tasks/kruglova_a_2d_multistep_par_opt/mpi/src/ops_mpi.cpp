#include "kruglova_a_2d_multistep_par_opt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "kruglova_a_2d_multistep_par_opt/common/include/common.hpp"

namespace kruglova_a_2d_multistep_par_opt {

namespace {

struct Trial1D {
  double x = 0.0;
  double z = 0.0;
};

struct Trial2D {
  double x = 0.0;
  double y = 0.0;
  double f = 0.0;
};

struct IntervalData {
  double x1 = 0.0, x2 = 0.0, f1 = 0.0, f2 = 0.0;
};

struct CharIdx {
  double r_val;
  size_t idx;
};

size_t FindBestIntervalIdx(const std::vector<Trial1D> &trials, double m_scaled) {
  double max_rate = -std::numeric_limits<double>::infinity();
  size_t best_idx = 0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    const double dz = trials[i + 1].z - trials[i].z;
    const double rate = (m_scaled * dx) + ((dz * dz) / (m_scaled * dx)) - (2.0 * (trials[i + 1].z + trials[i].z));
    if (rate > max_rate) {
      max_rate = rate;
      best_idx = i;
    }
  }
  return best_idx;
}

double CalculateM1D(const std::vector<Trial1D> &trials) {
  double m_max = 0.0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    if (dx > 1e-15) {
      const double dz = std::abs(trials[i + 1].z - trials[i].z);
      m_max = std::max(m_max, dz / dx);
    }
  }
  return m_max;
}

double Solve1DStrongin(const std::function<double(double)> &func, double a, double b, double eps, int max_iters,
                       double &best_x) {
  const double r_param = 2.0;
  std::vector<Trial1D> trials = {{a, func(a)}, {b, func(b)}};
  if (trials[0].x > trials[1].x) {
    std::swap(trials[0], trials[1]);
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    const double m_val = CalculateM1D(trials);
    const double m_scaled = (m_val > 0.0) ? (r_param * m_val) : 1.0;

    const size_t b_idx = FindBestIntervalIdx(trials, m_scaled);
    if ((trials[b_idx + 1].x - trials[b_idx].x) < eps) {
      break;
    }

    const double x_new =
        (0.5 * (trials[b_idx + 1].x + trials[b_idx].x)) - ((trials[b_idx + 1].z - trials[b_idx].z) / (2.0 * m_scaled));

    Trial1D new_trial{.x = x_new, .z = func(x_new)};
    auto it = std::lower_bound(trials.begin(), trials.end(), new_trial,
                               [](const Trial1D &t1, const Trial1D &t2) { return t1.x < t2.x; });
    trials.insert(it, new_trial);
  }

  auto best_it =
      std::min_element(trials.begin(), trials.end(), [](const Trial1D &t1, const Trial1D &t2) { return t1.z < t2.z; });
  best_x = best_it->x;
  return best_it->z;
}

void MasterCalculateIntervals(const std::vector<Trial2D> &trials, std::vector<IntervalData> &selected, int size,
                              double eps, int &stop_flag) {
  double m_max = 0.0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    m_max = std::max(m_max, std::abs(trials[i + 1].f - trials[i].f) / dx);
  }

  const double m_val = (m_max > 0.0) ? (2.0 * m_max) : 1.0;
  std::vector<CharIdx> rates;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    const double df = trials[i + 1].f - trials[i].f;
    const double r_val = (m_val * dx) + ((df * df) / (m_val * dx)) - (2.0 * (trials[i + 1].f + trials[i].f));
    rates.push_back({r_val, i});
  }

  std::sort(rates.begin(), rates.end(), [](const CharIdx &a, const CharIdx &b) { return a.r_val > b.r_val; });

  if (rates.empty() || (trials[rates[0].idx + 1].x - trials[rates[0].idx].x) < eps) {
    stop_flag = 1;
    return;
  }

  for (int i = 0; i < size; ++i) {
    const size_t s_idx = (static_cast<size_t>(i) < rates.size()) ? rates[static_cast<size_t>(i)].idx : rates[0].idx;
    selected[static_cast<size_t>(i)] = {
        .x1 = trials[s_idx].x, .x2 = trials[s_idx + 1].x, .f1 = trials[s_idx].f, .f2 = trials[s_idx + 1].f};
  }
}

std::vector<Trial2D> InitTrials(const InType &in, int init_points) {
  std::vector<Trial2D> trials;
  for (int i = 0; i < init_points; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(init_points - 1);
    const double x = in.x_min + ((in.x_max - in.x_min) * t);
    double y_best = 0.0;
    const double f = Solve1DStrongin([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                                     std::max(20, in.max_iters / 20), y_best);
    trials.push_back({.x = x, .y = y_best, .f = f});
  }
  std::sort(trials.begin(), trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
  return trials;
}

Trial2D ComputeLocalTrial(const IntervalData &interval, const InType &in) {
  const double dx = interval.x2 - interval.x1;
  double m_loc = std::abs(interval.f2 - interval.f1) / dx;
  m_loc = (m_loc > 0.0) ? (2.0 * m_loc) : 1.0;

  const double x_new = (0.5 * (interval.x1 + interval.x2)) - ((interval.f2 - interval.f1) / (2.0 * m_loc));
  double y_res = 0.0;
  const double f_res = Solve1DStrongin([&](double y) { return ObjectiveFunction(x_new, y); }, in.y_min, in.y_max,
                                       in.eps, std::max(25, in.max_iters / 20), y_res);
  return {.x = x_new, .y = y_res, .f = f_res};
}

void UpdateTrialsOnMaster(std::vector<Trial2D> &trials, const std::vector<Trial2D> &new_trials) {
  for (const auto &res : new_trials) {
    auto it = std::lower_bound(trials.begin(), trials.end(), res,
                               [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
    if (it == trials.end() || std::abs(it->x - res.x) > 1e-12) {
      trials.insert(it, res);
    }
  }
}

}  // namespace

KruglovaA2DMuitMPI::KruglovaA2DMuitMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KruglovaA2DMuitMPI::ValidationImpl() {
  const auto &in = GetInput();
  return in.x_max > in.x_min && in.y_max > in.y_min && in.eps > 0.0 && in.max_iters > 0;
}

bool KruglovaA2DMuitMPI::PreProcessingImpl() {
  GetOutput() = {0.0, 0.0, std::numeric_limits<double>::max()};
  return true;
}

bool KruglovaA2DMuitMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &in = GetInput();
  std::vector<Trial2D> trials;

  if (rank == 0) {
    trials = InitTrials(in, 20);
  }

  for (int iter = 0; iter < in.max_iters; ++iter) {
    int stop_flag = 0;
    std::vector<IntervalData> selected_intervals(static_cast<size_t>(size));

    if (rank == 0) {
      MasterCalculateIntervals(trials, selected_intervals, size, in.eps, stop_flag);
    }

    MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (stop_flag != 0) {
      break;
    }

    IntervalData my_interval{};
    MPI_Scatter(selected_intervals.data(), sizeof(IntervalData), MPI_BYTE, &my_interval, sizeof(IntervalData), MPI_BYTE,
                0, MPI_COMM_WORLD);

    Trial2D local_t = ComputeLocalTrial(my_interval, in);
    std::array<double, 3> send_res = {local_t.x, local_t.y, local_t.f};
    std::vector<double> recv_res(static_cast<size_t>(size) * 3);
    MPI_Gather(send_res.data(), 3, MPI_DOUBLE, recv_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      std::vector<Trial2D> next_trials;
      for (int i = 0; i < size; ++i) {
        size_t b = static_cast<size_t>(i) * 3;
        next_trials.push_back({.x = recv_res[b], .y = recv_res[b + 1], .f = recv_res[b + 2]});
      }
      UpdateTrialsOnMaster(trials, next_trials);
    }
  }

  std::array<double, 3> final_res = {0.0, 0.0, 0.0};
  if (rank == 0) {
    auto best_it =
        std::min_element(trials.begin(), trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.f < b.f; });
    final_res = {best_it->x, best_it->y, best_it->f};
  }

  MPI_Bcast(final_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = {final_res[0], final_res[1], final_res[2]};
  return true;
}

bool KruglovaA2DMuitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
