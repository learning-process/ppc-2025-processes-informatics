#include "kruglova_a_2d_multistep_par_opt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
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

double CalculateM1D(const std::vector<Trial1D> &trials) {
  double m_max = 0.0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    double dz = std::abs(trials[i + 1].z - trials[i].z);
    if (dx > 1e-15) {
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
    double m_val = CalculateM1D(trials);
    double m_scaled = (m_val > 0.0) ? (r_param * m_val) : 1.0;

    double max_rate = -std::numeric_limits<double>::infinity();
    size_t best_idx = 0;

    for (size_t i = 0; i + 1 < trials.size(); ++i) {
      double dx = trials[i + 1].x - trials[i].x;
      double dz = trials[i + 1].z - trials[i].z;
      double rate = (m_scaled * dx) + ((dz * dz) / (m_scaled * dx)) - (2.0 * (trials[i + 1].z + trials[i].z));
      if (rate > max_rate) {
        max_rate = rate;
        best_idx = i;
      }
    }

    if ((trials[best_idx + 1].x - trials[best_idx].x) < eps) {
      break;
    }

    double x_new = 0.5 * (trials[best_idx + 1].x + trials[best_idx].x) -
                   (trials[best_idx + 1].z - trials[best_idx].z) / (2.0 * m_scaled);

    Trial1D new_trial{.x = x_new, .z = func(x_new)};

    size_t pos = 0;
    while (pos < trials.size() && trials[pos].x < new_trial.x) {
      ++pos;
    }
    trials.insert(trials.begin() + static_cast<std::ptrdiff_t>(pos), new_trial);
  }

  size_t best_idx = 0;
  for (size_t i = 1; i < trials.size(); ++i) {
    if (trials[i].z < trials[best_idx].z) {
      best_idx = i;
    }
  }

  best_x = trials[best_idx].x;
  return trials[best_idx].z;
}

void MasterCalculateIntervals(const std::vector<Trial2D> &trials, std::vector<IntervalData> &selected, int size,
                              double eps, int &stop_flag) {
  double m_max = 0.0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    m_max = std::max(m_max, std::abs(trials[i + 1].f - trials[i].f) / dx);
  }

  double m_val = (m_max > 0.0) ? (2.0 * m_max) : 1.0;
  std::vector<CharIdx> rates;

  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    double df = trials[i + 1].f - trials[i].f;
    double r_val = (m_val * dx) + ((df * df) / (m_val * dx)) - (2.0 * (trials[i + 1].f + trials[i].f));
    rates.push_back({r_val, i});
  }

  for (size_t i = 0; i < rates.size(); ++i) {
    size_t max_idx = i;
    for (size_t j = i + 1; j < rates.size(); ++j) {
      if (rates[j].r_val > rates[max_idx].r_val) {
        max_idx = j;
      }
    }
    std::swap(rates[i], rates[max_idx]);
  }

  if (rates.empty() || (trials[rates[0].idx + 1].x - trials[rates[0].idx].x) < eps) {
    stop_flag = 1;
    return;
  }

  for (int i = 0; i < size; ++i) {
    size_t s_idx = (static_cast<size_t>(i) < rates.size()) ? rates[static_cast<size_t>(i)].idx : rates[0].idx;
    selected[static_cast<size_t>(i)] = {trials[s_idx].x, trials[s_idx + 1].x, trials[s_idx].f, trials[s_idx + 1].f};
  }
}

std::vector<Trial2D> InitTrials(const InType &in, int init_points) {
  std::vector<Trial2D> trials;

  for (int i = 0; i < init_points; ++i) {
    double x = in.x_min + (in.x_max - in.x_min) * static_cast<double>(i) / static_cast<double>(init_points - 1);

    double y_best = 0.0;
    double f = Solve1DStrongin([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                               std::max(20, in.max_iters / 20), y_best);

    trials.push_back({x, y_best, f});
  }

  for (size_t i = 0; i < trials.size(); ++i) {
    size_t min_idx = i;
    for (size_t j = i + 1; j < trials.size(); ++j) {
      if (trials[j].x < trials[min_idx].x) {
        min_idx = j;
      }
    }
    std::swap(trials[i], trials[min_idx]);
  }

  return trials;
}

Trial2D ComputeLocalTrial(const IntervalData &interval, const InType &in) {
  double dx = interval.x2 - interval.x1;
  double m = std::abs(interval.f2 - interval.f1) / dx;
  m = (m > 0.0) ? (2.0 * m) : 1.0;

  double x_new = 0.5 * (interval.x1 + interval.x2) - (interval.f2 - interval.f1) / (2.0 * m);

  double y_res = 0.0;
  double f_res = Solve1DStrongin([&](double y) { return ObjectiveFunction(x_new, y); }, in.y_min, in.y_max, in.eps,
                                 std::max(25, in.max_iters / 20), y_res);

  return {x_new, y_res, f_res};
}

void UpdateTrialsOnMaster(std::vector<Trial2D> &trials, const std::vector<Trial2D> &new_trials) {
  for (const auto &res : new_trials) {
    size_t pos = 0;
    while (pos < trials.size() && trials[pos].x < res.x) {
      ++pos;
    }
    if (pos == trials.size() || std::abs(trials[pos].x - res.x) > 1e-12) {
      trials.insert(trials.begin() + static_cast<std::ptrdiff_t>(pos), res);
    }
  }
}

Trial2D FindBestTrial(const std::vector<Trial2D> &trials) {
  size_t best_idx = 0;
  for (size_t i = 1; i < trials.size(); ++i) {
    if (trials[i].f < trials[best_idx].f) {
      best_idx = i;
    }
  }
  return trials[best_idx];
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

    Trial2D local_trial = ComputeLocalTrial(my_interval, in);
    std::array<double, 3> send_res = {local_trial.x, local_trial.y, local_trial.f};
    std::vector<double> recv_res(static_cast<size_t>(size) * 3);
    MPI_Gather(send_res.data(), 3, MPI_DOUBLE, recv_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      std::vector<Trial2D> new_trials;
      for (int i = 0; i < size; ++i) {
        size_t base = static_cast<size_t>(i) * 3;
        new_trials.push_back({.x = recv_res[base], .y = recv_res[base + 1], .f = recv_res[base + 2]});
      }
      UpdateTrialsOnMaster(trials, new_trials);
    }
  }

  std::array<double, 3> final_res = {0.0, 0.0, 0.0};
  if (rank == 0) {
    Trial2D best = FindBestTrial(trials);
    final_res = {best.x, best.y, best.f};
  }

  MPI_Bcast(final_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = {final_res[0], final_res[1], final_res[2]};
  return true;
}

bool KruglovaA2DMuitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
