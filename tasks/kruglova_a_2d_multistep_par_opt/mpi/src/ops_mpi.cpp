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
  double x;
  double z;
};

struct Trial2D {
  double x;
  double y;
  double z;
};

struct IntervalData {
  double x1;
  double x2;
  double f1;
  double f2;
};

struct CharIdx {
  double r_val;
  size_t idx;
};

template <typename T>
double CalculateM(const std::vector<T> &trials) {
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

  return best_idx;
}

size_t FindBestZ2D(const std::vector<Trial2D> &trials) {
  size_t best = 0;
  for (size_t i = 1; i < trials.size(); ++i) {
    if (trials[i].z < trials[best].z) {
      best = i;
    }
  }
  return best;
}

void InsertSorted1D(std::vector<Trial1D> &trials, const Trial1D &value) {
  auto it = std::lower_bound(trials.begin(), trials.end(), value,
                             [](const Trial1D &a, const Trial1D &b) { return a.x < b.x; });
  trials.insert(it, value);
}

void InsertSorted2D(std::vector<Trial2D> &trials, const Trial2D &value) {
  auto it = std::lower_bound(trials.begin(), trials.end(), value,
                             [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
  if (it == trials.end() || std::abs(it->x - value.x) > 1e-12) {
    trials.insert(it, value);
  }
}

double Solve1DStrongin(const std::function<double(double)> &func, double a, double b, double eps, int max_iters,
                       double &best_x) {
  const double r_param = 2.0;

  std::vector<Trial1D> trials{{a, func(a)}, {b, func(b)}};
  if (trials[0].x > trials[1].x) {
    std::swap(trials[0], trials[1]);
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    double m_val = CalculateM(trials);
    double m_scaled = (m_val > 0.0) ? (r_param * m_val) : 1.0;

    size_t idx = FindBestInterval1D(trials, m_scaled);
    double dx = trials[idx + 1].x - trials[idx].x;
    if (dx < eps) {
      break;
    }

    double x_new = 0.5 * (trials[idx + 1].x + trials[idx].x) - ((trials[idx + 1].z - trials[idx].z) / (2.0 * m_scaled));
    InsertSorted1D(trials, Trial1D{x_new, func(x_new)});
  }

  size_t best = FindBestZ1D(trials);
  best_x = trials[best].x;
  return trials[best].z;
}

void MasterCalculateIntervals(const std::vector<Trial2D> &trials, std::vector<IntervalData> &selected, int size,
                              double eps, int &stop_flag) {
  double m_max = 0.0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    m_max = std::max(m_max, std::abs(trials[i + 1].z - trials[i].z) / dx);
  }

  double m_val = (m_max > 0.0) ? (2.0 * m_max) : 1.0;
  std::vector<CharIdx> rates;

  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    double df = trials[i + 1].z - trials[i].z;
    double r_val = (m_val * dx) + ((df * df) / (m_val * dx)) - (2.0 * (trials[i + 1].z + trials[i].z));
    rates.push_back(CharIdx{r_val, i});
  }

  std::sort(rates.begin(), rates.end(), [](const CharIdx &a, const CharIdx &b) { return a.r_val > b.r_val; });

  if (rates.empty() || (trials[rates[0].idx + 1].x - trials[rates[0].idx].x) < eps) {
    stop_flag = 1;
    return;
  }

  for (int i = 0; i < size; ++i) {
    size_t s_idx = (static_cast<size_t>(i) < rates.size()) ? rates[i].idx : rates[0].idx;
    selected[i] = IntervalData{trials[s_idx].x, trials[s_idx + 1].x, trials[s_idx].z, trials[s_idx + 1].z};
  }
}

std::vector<Trial2D> InitTrials(const InType &in, int init_points) {
  std::vector<Trial2D> trials;
  for (int i = 0; i < init_points; ++i) {
    double x = in.x_min + (in.x_max - in.x_min) * i / static_cast<double>(init_points - 1);
    double y_best = 0.0;
    double z = Solve1DStrongin([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                               std::max(20, in.max_iters / 20), y_best);
    trials.push_back(Trial2D{x, y_best, z});
  }

  std::sort(trials.begin(), trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
  return trials;
}

Trial2D ComputeLocalTrial(const IntervalData &interval, const InType &in) {
  double dx = interval.x2 - interval.x1;
  double m = std::abs(interval.f2 - interval.f1) / dx;
  m = (m > 0.0) ? (2.0 * m) : 1.0;

  double x_new = 0.5 * (interval.x1 + interval.x2) - ((interval.f2 - interval.f1) / (2.0 * m));
  double y_res = 0.0;
  double f_res = Solve1DStrongin([&](double y) { return ObjectiveFunction(x_new, y); }, in.y_min, in.y_max, in.eps,
                                 std::max(25, in.max_iters / 20), y_res);
  return Trial2D{x_new, y_res, f_res};
}

void UpdateTrialsOnMaster(std::vector<Trial2D> &trials, const std::vector<Trial2D> &new_trials) {
  for (const auto &res : new_trials) {
    InsertSorted2D(trials, res);
  }
}

Trial2D FindBestTrial(const std::vector<Trial2D> &trials) {
  return *std::min_element(trials.begin(), trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.z < b.z; });
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
    std::array<double, 3> send_res{local_trial.x, local_trial.y, local_trial.z};
    std::vector<double> recv_res(static_cast<size_t>(size) * 3);

    MPI_Gather(send_res.data(), 3, MPI_DOUBLE, recv_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      std::vector<Trial2D> new_trials;
      for (int i = 0; i < size; ++i) {
        size_t base = static_cast<size_t>(i) * 3;
        new_trials.push_back(Trial2D{recv_res[base], recv_res[base + 1], recv_res[base + 2]});
      }
      UpdateTrialsOnMaster(trials, new_trials);
    }
  }

  std::array<double, 3> final_res{0.0, 0.0, 0.0};
  if (rank == 0) {
    Trial2D best = FindBestTrial(trials);
    final_res = {best.x, best.y, best.z};
  }

  MPI_Bcast(final_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = {final_res[0], final_res[1], final_res[2]};
  return true;
}

bool KruglovaA2DMuitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
