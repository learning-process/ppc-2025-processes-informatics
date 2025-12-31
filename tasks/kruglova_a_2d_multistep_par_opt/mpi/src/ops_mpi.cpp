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
  Trial1D(double x_val, double z_val) : x(x_val), z(z_val) {}
};

struct Trial2D {
  double x;
  double y;
  double f;
  Trial2D(double x_val, double y_val, double f_val) : x(x_val), y(y_val), f(f_val) {}
};

struct CharIdx {
  double r_val;
  size_t idx;
};

double CalculateM1D(const std::vector<Trial1D> &trials) {
  double m_max = 0.0;
  for (size_t i = 0; (i + 1) < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    if (dx > 1e-15) {
      const double dz = std::abs(trials[i + 1].z - trials[i].z);
      const double ratio = dz / dx;
      if (ratio > m_max) {
        m_max = ratio;
      }
    }
  }
  return m_max;
}

double CalculateM2D(const std::vector<Trial2D> &trials) {
  double m_max = 0.0;
  for (size_t i = 0; (i + 1) < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    if (dx > 1e-15) {
      const double df = std::abs(trials[i + 1].f - trials[i].f);
      const double ratio = df / dx;
      if (ratio > m_max) {
        m_max = ratio;
      }
    }
  }
  return m_max;
}

void InsertSorted1D(std::vector<Trial1D> &trials, const Trial1D &value) {
  size_t pos = 0;
  while (pos < trials.size() && trials[pos].x < value.x) {
    ++pos;
  }
  trials.insert(trials.begin() + static_cast<std::ptrdiff_t>(pos), value);
}

void InsertSorted2D(std::vector<Trial2D> &trials, const Trial2D &value) {
  size_t pos = 0;
  while (pos < trials.size() && trials[pos].x < value.x) {
    ++pos;
  }
  if (pos == trials.size() || (std::abs(trials[pos].x - value.x) > 1e-12)) {
    trials.insert(trials.begin() + static_cast<std::ptrdiff_t>(pos), value);
  }
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
    const double m_val = CalculateM1D(trials);
    const double m_scaled = (m_val > 0.0) ? (r_param * m_val) : 1.0;

    double max_rate = -std::numeric_limits<double>::infinity();
    size_t idx = 0;
    for (size_t i = 0; (i + 1) < trials.size(); ++i) {
      const double dx = trials[i + 1].x - trials[i].x;
      const double dz = trials[i + 1].z - trials[i].z;
      const double rate = (m_scaled * dx) + ((dz * dz) / (m_scaled * dx)) - (2.0 * (trials[i + 1].z + trials[i].z));
      if (rate > max_rate) {
        max_rate = rate;
        idx = i;
      }
    }

    if ((trials[idx + 1].x - trials[idx].x) < eps) {
      break;
    }

    const double x_new =
        (0.5 * (trials[idx + 1].x + trials[idx].x)) - ((trials[idx + 1].z - trials[idx].z) / (2.0 * m_scaled));
    InsertSorted1D(trials, Trial1D(x_new, func(x_new)));
  }

  size_t best = 0;
  for (size_t i = 1; i < trials.size(); ++i) {
    if (trials[i].z < trials[best].z) {
      best = i;
    }
  }
  best_x = trials[best].x;
  return trials[best].z;
}

void PrepareIntervals(const std::vector<Trial2D> &trials, std::vector<double> &buf, int size, double eps, int &stop) {
  const double m_max = CalculateM2D(trials);
  const double m_v = (m_max > 0.0) ? (2.0 * m_max) : 1.0;
  std::vector<CharIdx> rates;
  for (size_t i = 0; (i + 1) < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    const double df = trials[i + 1].f - trials[i].f;
    const double r = (m_v * dx) + ((df * df) / (m_v * dx)) - (2.0 * (trials[i + 1].f + trials[i].f));
    rates.push_back({r, i});
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

  if (rates.empty() || (trials[rates[0].idx + 1].x - trials[rates[0].idx].x < eps)) {
    stop = 1;
  } else {
    for (int i = 0; i < size; ++i) {
      const size_t s_idx = (static_cast<size_t>(i) < rates.size()) ? rates[static_cast<size_t>(i)].idx : rates[0].idx;
      buf[(static_cast<size_t>(i) * 4) + 0] = trials[s_idx].x;
      buf[(static_cast<size_t>(i) * 4) + 1] = trials[s_idx + 1].x;
      buf[(static_cast<size_t>(i) * 4) + 2] = trials[s_idx].f;
      buf[(static_cast<size_t>(i) * 4) + 3] = trials[s_idx + 1].f;
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
  return (in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.eps > 0.0) && (in.max_iters > 0);
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
    for (int i = 0; i < 15; ++i) {
      const double x = in.x_min + ((in.x_max - in.x_min) * (static_cast<double>(i) / 14.0));
      double y_b = 0.0;
      const double f = Solve1DStrongin([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                                       std::max(40, in.max_iters / 15), y_b);
      InsertSorted2D(trials, Trial2D(x, y_b, f));
    }
  }

  for (int iter = 0; iter < in.max_iters; ++iter) {
    int stop_f = 0;
    std::vector<double> intervals_buf(static_cast<size_t>(size) * 4, 0.0);
    if (rank == 0) {
      PrepareIntervals(trials, intervals_buf, size, in.eps, stop_f);
    }

    MPI_Bcast(&stop_f, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (stop_f != 0) {
      break;
    }

    std::array<double, 4> my_int{};
    MPI_Scatter(intervals_buf.data(), 4, MPI_DOUBLE, my_int.data(), 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double dx_l = my_int[1] - my_int[0];
    const double m_l = std::max(1.0, 2.0 * (std::abs(my_int[3] - my_int[2]) / dx_l));
    const double x_n = (0.5 * (my_int[0] + my_int[1])) - ((my_int[3] - my_int[2]) / (2.0 * m_l));
    double y_res = 0.0;
    const double f_res = Solve1DStrongin([&](double y) { return ObjectiveFunction(x_n, y); }, in.y_min, in.y_max,
                                         in.eps, std::max(50, in.max_iters / 15), y_res);

    std::array<double, 3> send_v = {x_n, y_res, f_res};
    std::vector<double> recv_v(static_cast<size_t>(size) * 3);
    MPI_Gather(send_v.data(), 3, MPI_DOUBLE, recv_v.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      for (int i = 0; i < size; ++i) {
        const size_t base = static_cast<size_t>(i) * 3;
        InsertSorted2D(trials, Trial2D(recv_v[base + 0], recv_v[base + 1], recv_v[base + 2]));
      }
    }
  }

  std::array<double, 3> res = {0.0, 0.0, 0.0};
  if (rank == 0) {
    size_t best_idx = 0;
    for (size_t i = 1; i < trials.size(); ++i) {
      if (trials[i].f < trials[best_idx].f) {
        best_idx = i;
      }
    }
    res = {trials[best_idx].x, trials[best_idx].y, trials[best_idx].f};
  }
  MPI_Bcast(res.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = {res[0], res[1], res[2]};
  return true;
}

bool KruglovaA2DMuitMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
