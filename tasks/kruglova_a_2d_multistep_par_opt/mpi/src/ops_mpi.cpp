#include "kruglova_a_2d_multistep_par_opt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
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

double Solve1DStrongin(std::function<double(double)> func, double a, double b, double eps, int max_iters,
                       double &best_x) {
  const double r = 2.0;
  std::vector<Trial1D> trials = {{a, func(a)}, {b, func(b)}};
  if (trials[0].x > trials[1].x) {
    std::swap(trials[0], trials[1]);
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    double M = 0.0;
    for (size_t i = 0; i + 1 < trials.size(); ++i) {
      double dx = trials[i + 1].x - trials[i].x;
      double dz = std::abs(trials[i + 1].z - trials[i].z);
      if (dx > 1e-15) {
        M = std::max(M, dz / dx);
      }
    }

    double m = (M > 0.0) ? r * M : 1.0;
    double max_R = -std::numeric_limits<double>::infinity();
    size_t best_idx = 0;

    for (size_t i = 0; i + 1 < trials.size(); ++i) {
      double dx = trials[i + 1].x - trials[i].x;
      double dz = trials[i + 1].z - trials[i].z;
      double R = m * dx + (dz * dz) / (m * dx) - 2.0 * (trials[i + 1].z + trials[i].z);
      if (R > max_R) {
        max_R = R;
        best_idx = i;
      }
    }

    if (trials[best_idx + 1].x - trials[best_idx].x < eps) {
      break;
    }

    double x_new =
        0.5 * (trials[best_idx + 1].x + trials[best_idx].x) - (trials[best_idx + 1].z - trials[best_idx].z) / (2.0 * m);

    Trial1D new_trial = {x_new, func(x_new)};

    auto it = std::lower_bound(trials.begin(), trials.end(), new_trial,
                               [](const Trial1D &t1, const Trial1D &t2) { return t1.x < t2.x; });
    trials.insert(it, new_trial);
  }

  auto best =
      std::min_element(trials.begin(), trials.end(), [](const Trial1D &a, const Trial1D &b) { return a.z < b.z; });
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
  GetOutput() = {0.0, 0.0, std::numeric_limits<double>::max()};
  return true;
}

bool KruglovaA2DMuitMPI::RunImpl() {
  int rank, size;
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
      double x = in.x_min + (in.x_max - in.x_min) * i / (double)(init_points - 1);
      double y_best;
      double f = Solve1DStrongin([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                                 std::max(50, in.max_iters / 10), y_best);
      trials.push_back({x, y_best, f});
    }
    std::sort(trials.begin(), trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
  }

  for (int iter = 0; iter < in.max_iters; ++iter) {
    int stop_flag = 0;

    struct IntervalData {
      double x1, x2, f1, f2;
    };
    std::vector<IntervalData> selected_intervals(size);

    if (rank == 0) {
      struct CharIdx {
        double R;
        size_t idx;
      };
      std::vector<CharIdx> rates;

      double M = 0.0;
      for (size_t i = 0; i + 1 < trials.size(); ++i) {
        double dx = trials[i + 1].x - trials[i].x;
        M = std::max(M, std::abs(trials[i + 1].f - trials[i].f) / dx);
      }
      double m = (M > 0.0) ? 2.0 * M : 1.0;

      for (size_t i = 0; i + 1 < trials.size(); ++i) {
        double dx = trials[i + 1].x - trials[i].x;
        double df = trials[i + 1].f - trials[i].f;
        double R = m * dx + (df * df) / (m * dx) - 2.0 * (trials[i + 1].f + trials[i].f);
        rates.push_back({R, i});
      }

      std::sort(rates.begin(), rates.end(), [](const CharIdx &a, const CharIdx &b) { return a.R > b.R; });

      if (trials[rates[0].idx + 1].x - trials[rates[0].idx].x < in.eps) {
        stop_flag = 1;
      } else {
        for (int i = 0; i < size; ++i) {
          size_t idx = (i < (int)rates.size()) ? rates[i].idx : rates[0].idx;
          selected_intervals[i] = {trials[idx].x, trials[idx + 1].x, trials[idx].f, trials[idx + 1].f};
        }
      }
    }

    MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (stop_flag) {
      break;
    }

    IntervalData my_interval;
    MPI_Scatter(selected_intervals.data(), sizeof(IntervalData), MPI_BYTE, &my_interval, sizeof(IntervalData), MPI_BYTE,
                0, MPI_COMM_WORLD);

    double M_local = std::abs(my_interval.f2 - my_interval.f1) / (my_interval.x2 - my_interval.x1);
    double m_local = (M_local > 0.0) ? 2.0 * M_local : 1.0;

    double x_new = 0.5 * (my_interval.x1 + my_interval.x2) - (my_interval.f2 - my_interval.f1) / (2.0 * m_local);

    double y_res;
    double f_res = Solve1DStrongin([&](double y) { return ObjectiveFunction(x_new, y); }, in.y_min, in.y_max, in.eps,
                                   std::max(50, in.max_iters / 10), y_res);

    double send_res[3] = {x_new, y_res, f_res};
    std::vector<double> recv_res(size * 3);
    MPI_Gather(send_res, 3, MPI_DOUBLE, recv_res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      for (int i = 0; i < size; ++i) {
        Trial2D res = {recv_res[i * 3], recv_res[i * 3 + 1], recv_res[i * 3 + 2]};
        auto it = std::lower_bound(trials.begin(), trials.end(), res,
                                   [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });
        if (it == trials.end() || std::abs(it->x - res.x) > 1e-12) {
          trials.insert(it, res);
        }
      }
    }
  }

  double final_res[3] = {0.0, 0.0, 0.0};
  if (rank == 0) {
    auto best_it =
        std::min_element(trials.begin(), trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.f < b.f; });
    final_res[0] = best_it->x;
    final_res[1] = best_it->y;
    final_res[2] = best_it->f;
  }

  MPI_Bcast(final_res, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = {final_res[0], final_res[1], final_res[2]};

  return true;
}

bool KruglovaA2DMuitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
