#include "kruglova_a_2d_multistep_par_opt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <ranges>
#include <vector>

#include "kruglova_a_2d_multistep_par_opt/common/include/common.hpp"

namespace kruglova_a_2d_multistep_par_opt {

namespace {

struct Trial1D {
  double x;
  double z;
};

double CalculateM(const std::vector<Trial1D> &trials, double r) {
  double max_m = 0.0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    double dx = trials[i + 1].x - trials[i].x;
    if (dx > 1e-15) {
      double cur_m = std::abs(trials[i + 1].z - trials[i].z) / dx;
      if (cur_m > max_m) {
        max_m = cur_m;
      }
    }
  }
  return (max_m > 0.0) ? r * max_m : 1.0;
}

double Solve1DStrongin(const std::function<double(double)> &func, double a, double b, double eps, int max_iters,
                       double &best_x) {
  const double r = 2.0;
  std::vector<Trial1D> trials = {{a, func(a)}, {b, func(b)}};
  if (trials[0].x > trials[1].x) {
    std::swap(trials[0], trials[1]);
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    double m = CalculateM(trials, r);
    double max_r = -std::numeric_limits<double>::infinity();
    size_t best_idx = 0;

    for (size_t i = 0; i + 1 < trials.size(); ++i) {
      double dx = trials[i + 1].x - trials[i].x;
      double dz = trials[i + 1].z - trials[i].z;
      double r_val = (m * dx) + ((dz * dz) / (m * dx)) - (2.0 * (trials[i + 1].z + trials[i].z));
      if (r_val > max_r) {
        max_r = r_val;
        best_idx = i;
      }
    }

    if (trials[best_idx + 1].x - trials[best_idx].x < eps) {
      break;
    }

    double x_new = 0.5 * (trials[best_idx + 1].x + trials[best_idx].x) -
                   ((trials[best_idx + 1].z - trials[best_idx].z) / (2.0 * m));

    Trial1D new_trial = {x_new, func(x_new)};

    auto it =
        std::ranges::lower_bound(trials, new_trial, [](const Trial1D &t1, const Trial1D &t2) { return t1.x < t2.x; });
    trials.insert(it, new_trial);
  }

  auto best = std::ranges::min_element(trials, [](const Trial1D &a, const Trial1D &b) { return a.z < b.z; });
  best_x = best->x;
  return best->z;
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
  GetOutput() = {.x = 0.0, .y = 0.0, .f_value = std::numeric_limits<double>::max()};
  return true;
}

bool KruglovaA2DMuitMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &in = GetInput();

  struct Trial2D {
    double x, y, f;
  };
  std::vector<Trial2D> trials;

  if (rank == 0) {
    const int init_points = 40;
    for (int i = 0; i < init_points; ++i) {
      double x = in.x_min + ((in.x_max - in.x_min) * static_cast<double>(i) / (init_points - 1));
      double y_best = std::numeric_limits<double>::quiet_NaN();
      double f = Solve1DStrongin([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                                 std::max(50, in.max_iters / 10), y_best);
      trials.push_back({x, y_best, f});
    }
    std::ranges::sort(trials, [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
  }

  for (int iter = 0; iter < in.max_iters; ++iter) {
    int stop_flag = 0;

    struct IntervalData {
      double x1, x2, f1, f2;
    };
    std::vector<IntervalData> selected_intervals(static_cast<size_t>(size));

    if (rank == 0) {
      struct CharIdx {
        double r;
        size_t idx;
      };
      std::vector<CharIdx> rates;

      double m = 0.0;
      for (size_t i = 0; i + 1 < trials.size(); ++i) {
        double dx = trials[i + 1].x - trials[i].x;
        m = std::max(m, std::abs(trials[i + 1].f - trials[i].f) / dx);
      }
      double m_local = (m > 0.0) ? 2.0 * m : 1.0;

      for (size_t i = 0; i + 1 < trials.size(); ++i) {
        double dx = trials[i + 1].x - trials[i].x;
        double df = trials[i + 1].f - trials[i].f;
        double r = (m_local * dx) + ((df * df) / (m_local * dx)) - (2.0 * (trials[i + 1].f + trials[i].f));
        rates.push_back({r, i});
      }

      std::ranges::sort(rates, [](const CharIdx &a, const CharIdx &b) { return a.r > b.r; });

      if (rates.empty()) {
        stop_flag = 1;
      } else {
        size_t idx0 = rates[0].idx;
        if (idx0 + 1 < trials.size() && trials[idx0 + 1].x - trials[idx0].x < in.eps) {
          stop_flag = 1;
        } else {
          for (int i = 0; i < size; ++i) {
            size_t idx = (std::cmp_less(i, static_cast<int>(rates.size()))) ? rates[i].idx : idx0;
            selected_intervals[i] = {
                .x1 = trials[idx].x, .x2 = trials[idx + 1].x, .f1 = trials[idx].f, .f2 = trials[idx + 1].f};
          }
        }
      }
    }

    MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (stop_flag != 0) {
      break;
    }

    IntervalData my_interval = {};
    MPI_Scatter(selected_intervals.data(), sizeof(IntervalData), MPI_BYTE, &my_interval, sizeof(IntervalData), MPI_BYTE,
                0, MPI_COMM_WORLD);

    double m_local = std::abs(my_interval.f2 - my_interval.f1) / (my_interval.x2 - my_interval.x1);
    double m_local_val = (m_local > 0.0) ? 2.0 * m_local : 1.0;

    double x_new = 0.5 * (my_interval.x1 + my_interval.x2) - ((my_interval.f2 - my_interval.f1) / (2.0 * m_local_val));

    double y_res = std::numeric_limits<double>::quiet_NaN();
    double f_res = Solve1DStrongin([&](double y) { return ObjectiveFunction(x_new, y); }, in.y_min, in.y_max, in.eps,
                                   std::max(50, in.max_iters / 10), y_res);

    std::array<double, 3> send_res = {x_new, y_res, f_res};
    std::vector<double> recv_res(static_cast<size_t>(size) * 3);
    MPI_Gather(send_res.data(), 3, MPI_DOUBLE, recv_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      for (int i = 0; i < size; ++i) {
        Trial2D res = {.x = recv_res[(i * 3)], .y = recv_res[(i * 3) + 1], .f = recv_res[(i * 3) + 2]};
        auto it = std::ranges::lower_bound(trials, res, [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
        if (it == trials.end() || std::abs(it->x - res.x) > 1e-12) {
          trials.insert(it, res);
        }
      }
    }
  }

  std::array<double, 3> final_res = {0.0, 0.0, 0.0};
  if (rank == 0) {
    auto best_it = std::ranges::min_element(trials, [](const Trial2D &a, const Trial2D &b) { return a.f < b.f; });
    final_res[0] = best_it->x;
    final_res[1] = best_it->y;
    final_res[2] = best_it->f;
  }

  MPI_Bcast(final_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = {.x = final_res[0], .y = final_res[1], .f_value = final_res[2]};

  return true;
}

bool KruglovaA2DMuitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
