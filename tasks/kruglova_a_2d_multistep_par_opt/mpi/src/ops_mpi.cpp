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
  for (size_t i = 0; (i + 1) < trials.size(); ++i) {
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

    for (size_t i = 0; (i + 1) < trials.size(); ++i) {
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

    double x_new = (0.5 * (trials[best_idx + 1].x + trials[best_idx].x)) -
                   ((trials[best_idx + 1].z - trials[best_idx].z) / (2.0 * m_scaled));

    Trial1D new_trial{.x = x_new, .z = func(x_new)};
    auto it = std::lower_bound(trials.begin(), trials.end(), new_trial,
                               [](const Trial1D &t1, const Trial1D &t2) { return t1.x < t2.x; });
    trials.insert(it, new_trial);
  }

  auto best =
      std::min_element(trials.begin(), trials.end(), [](const Trial1D &t1, const Trial1D &t2) { return t1.z < t2.z; });
  best_x = best->x;
  return best->z;
}

void MasterCalculateIntervals(const std::vector<Trial2D> &trials, std::vector<IntervalData> &selected, int size,
                              double eps, int &stop_flag) {
  double m_max = 0.0;
  for (size_t i = 0; (i + 1) < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    m_max = std::max(m_max, std::abs(trials[i + 1].f - trials[i].f) / dx);
  }

  double m_val = (m_max > 0.0) ? (2.0 * m_max) : 1.0;
  std::vector<CharIdx> rates;
  for (size_t i = 0; (i + 1) < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    double df = trials[i + 1].f - trials[i].f;
    double r_val = (m_val * dx) + ((df * df) / (m_val * dx)) - (2.0 * (trials[i + 1].f + trials[i].f));
    rates.push_back({r_val, i});
  }

  std::sort(rates.begin(), rates.end(), [](const CharIdx &a, const CharIdx &b) { return a.r_val > b.r_val; });

  if (rates.empty() || ((trials[rates[0].idx + 1].x - trials[rates[0].idx].x) < eps)) {
    stop_flag = 1;
  } else {
    for (int i = 0; i < size; ++i) {
      size_t s_idx = (static_cast<size_t>(i) < rates.size()) ? rates[static_cast<size_t>(i)].idx : rates[0].idx;
      selected[static_cast<size_t>(i)] = {
          .x1 = trials[s_idx].x, .x2 = trials[s_idx + 1].x, .f1 = trials[s_idx].f, .f2 = trials[s_idx + 1].f};
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
    const int init_points = 20;
    for (int i = 0; i < init_points; ++i) {
      double x = in.x_min + (in.x_max - in.x_min) * static_cast<double>(i) / static_cast<double>(init_points - 1);
      double y_best = 0.0;
      double f = Solve1DStrongin([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                                 std::max(20, in.max_iters / 20), y_best);
      trials.push_back({.x = x, .y = y_best, .f = f});
    }
    std::sort(trials.begin(), trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
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

    double dx_local = my_interval.x2 - my_interval.x1;
    double m_local = (std::abs(my_interval.f2 - my_interval.f1) / dx_local);
    m_local = (m_local > 0.0) ? (2.0 * m_local) : 1.0;

    double x_new = (0.5 * (my_interval.x1 + my_interval.x2)) - ((my_interval.f2 - my_interval.f1) / (2.0 * m_local));
    double y_res = 0.0;
    double f_res = Solve1DStrongin([&](double y) { return ObjectiveFunction(x_new, y); }, in.y_min, in.y_max, in.eps,
                                   std::max(25, in.max_iters / 20), y_res);

    std::array<double, 3> send_res = {x_new, y_res, f_res};
    std::vector<double> recv_res(static_cast<size_t>(size) * 3);
    MPI_Gather(send_res.data(), 3, MPI_DOUBLE, recv_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      for (int i = 0; i < size; ++i) {
        size_t base = static_cast<size_t>(i) * 3;
        Trial2D res{.x = recv_res[base], .y = recv_res[base + 1], .f = recv_res[base + 2]};
        auto it = std::lower_bound(trials.begin(), trials.end(), res,
                                   [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
        if (it == trials.end() || std::abs(it->x - res.x) > 1e-12) {
          trials.insert(it, res);
        }
      }
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
