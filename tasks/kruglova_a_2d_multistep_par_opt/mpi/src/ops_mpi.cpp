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
  Trial1D(double x_val, double z_val) : x(x_val), z(z_val) {}
};

struct Trial2D {
  double x = 0.0;
  double y = 0.0;
  double f = 0.0;
  Trial2D(double x_val, double y_val, double f_val) : x(x_val), y(y_val), f(f_val) {}
};

struct IntervalData {
  double x1 = 0.0;
  double x2 = 0.0;
  double f1 = 0.0;
  double f2 = 0.0;
  IntervalData() = default;
  IntervalData(double x1v, double x2v, double f1v, double f2v) : x1(x1v), x2(x2v), f1(f1v), f2(f2v) {}
};

struct CharIdx {
  double r_val;
  size_t idx;
};

double CalculateM1D(const std::vector<Trial1D> &trials) {
  double m_max = 0.0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    if (dx > 1e-15) {
      double dz = std::abs(trials[i + 1].z - trials[i].z);
      m_max = std::max(m_max, dz / dx);
    }
  }
  return m_max;
}

size_t FindBestInterval1D(const std::vector<Trial1D> &trials, double m_scaled) {
  size_t best_idx = 0;
  double max_rate = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    double dz = trials[i + 1].z - trials[i].z;
    double rate = (m_scaled * dx) + ((dz * dz) / (m_scaled * dx)) - (2.0 * (trials[i + 1].z + trials[i].z));
    if (rate > max_rate) {
      max_rate = rate;
      best_idx = i;
    }
  }
  return best_idx;
}

void insert_sorted_1d(std::vector<Trial1D> &trials, const Trial1D &t) {
  auto it = std::find_if(trials.begin(), trials.end(), [&](const Trial1D &val) { return val.x >= t.x; });
  trials.insert(it, t);
}

double Solve1DStrongin(const std::function<double(double)> &func, double a, double b, double eps, int max_iters,
                       double &best_x) {
  const double r_param = 2.0;
  std::vector<Trial1D> trials;
  trials.emplace_back(a, func(a));
  trials.emplace_back(b, func(b));

  if (trials[0].x > trials[1].x) {
    std::swap(trials[0], trials[1]);
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    double m_val = CalculateM1D(trials);
    double m_scaled = (m_val > 0.0) ? (r_param * m_val) : 1.0;

    size_t best_idx = FindBestInterval1D(trials, m_scaled);
    double dx = trials[best_idx + 1].x - trials[best_idx].x;

    if (dx < eps) {
      break;
    }

    double x_new = (0.5 * (trials[best_idx + 1].x + trials[best_idx].x)) -
                   ((trials[best_idx + 1].z - trials[best_idx].z) / (2.0 * m_scaled));

    insert_sorted_1d(trials, Trial1D(x_new, func(x_new)));
  }

  auto best_it = std::min_element(trials.begin(), trials.end(),
                                  [](const Trial1D &left, const Trial1D &right) { return left.z < right.z; });
  best_x = best_it->x;
  return best_it->z;
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
  rates.reserve(trials.size());

  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    double df = trials[i + 1].f - trials[i].f;
    double r_val = (m_val * dx) + ((df * df) / (m_val * dx)) - (2.0 * (trials[i + 1].f + trials[i].f));
    rates.push_back({r_val, i});
  }

  std::sort(rates.begin(), rates.end(), [](const CharIdx &a, const CharIdx &b) { return a.r_val > b.r_val; });

  if (rates.empty() || (trials[rates[0].idx + 1].x - trials[rates[0].idx].x) < eps) {
    stop_flag = 1;
    return;
  }

  for (int i = 0; i < size; ++i) {
    size_t s_idx = (static_cast<size_t>(i) < rates.size()) ? rates[static_cast<size_t>(i)].idx : rates[0].idx;
    selected[static_cast<size_t>(i)] =
        IntervalData(trials[s_idx].x, trials[s_idx + 1].x, trials[s_idx].f, trials[s_idx + 1].f);
  }
}

std::vector<Trial2D> InitTrials(const InType &in, int init_points) {
  std::vector<Trial2D> trials;
  trials.reserve(static_cast<size_t>(init_points));

  for (int i = 0; i < init_points; ++i) {
    double x = in.x_min + ((in.x_max - in.x_min) * static_cast<double>(i) / static_cast<double>(init_points - 1));
    double y_best = 0.0;
    double f = Solve1DStrongin([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                               std::max(20, in.max_iters / 20), y_best);
    trials.emplace_back(x, y_best, f);
  }

  std::sort(trials.begin(), trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
  return trials;
}

Trial2D ComputeLocalTrial(const IntervalData &interval, const InType &in) {
  double dx = interval.x2 - interval.x1;
  double m = std::abs(interval.f2 - interval.f1) / dx;
  m = (m > 0.0) ? (2.0 * m) : 1.0;

  double x_new = (0.5 * (interval.x1 + interval.x2)) - ((interval.f2 - interval.f1) / (2.0 * m));

  double y_res = 0.0;
  double f_res = Solve1DStrongin([&](double y) { return ObjectiveFunction(x_new, y); }, in.y_min, in.y_max, in.eps,
                                 std::max(25, in.max_iters / 20), y_res);
  return Trial2D(x_new, y_res, f_res);
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

    IntervalData my_interval;
    MPI_Scatter(selected_intervals.data(), sizeof(IntervalData), MPI_BYTE, &my_interval, sizeof(IntervalData), MPI_BYTE,
                0, MPI_COMM_WORLD);

    Trial2D local_trial = ComputeLocalTrial(my_interval, in);
    std::array<double, 3> send_res = {local_trial.x, local_trial.y, local_trial.f};
    std::vector<double> recv_res(static_cast<size_t>(size) * 3);
    MPI_Gather(send_res.data(), 3, MPI_DOUBLE, recv_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      std::vector<Trial2D> new_trials;
      new_trials.reserve(static_cast<size_t>(size));
      for (int i = 0; i < size; ++i) {
        size_t base = static_cast<size_t>(i) * 3;
        new_trials.emplace_back(recv_res[base], recv_res[base + 1], recv_res[base + 2]);
      }
      UpdateTrialsOnMaster(trials, new_trials);
    }
  }

  std::array<double, 3> final_res{0.0, 0.0, 0.0};
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
