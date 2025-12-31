#include "kruglova_a_2d_multistep_par_opt/seq/include/ops_seq.hpp"

#include <algorithm>
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
  double z = 0.0;
};

template <typename T>
double CalculateM(const std::vector<T> &trials) {
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

size_t FindBestInterval(const std::vector<Trial2D> &trials, double m_scaled) {
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

double Solve1DStrongin(const std::function<double(double)> &func, double a, double b, double eps, int max_iters,
                       double &best_x) {
  const double r_param = 2.0;

  std::vector<Trial1D> trials;
  trials.push_back({a, func(a)});
  trials.push_back({b, func(b)});

  if (trials[0].x > trials[1].x) {
    std::swap(trials[0], trials[1]);
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    double m_val = CalculateM(trials);
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

    Trial1D new_trial{x_new, func(x_new)};

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

}  // namespace

KruglovaA2DMuitSEQ::KruglovaA2DMuitSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KruglovaA2DMuitSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.x_max > in.x_min && in.y_max > in.y_min && in.eps > 0.0 && in.max_iters > 0 && std::isfinite(in.x_min) &&
         std::isfinite(in.x_max) && std::isfinite(in.y_min) && std::isfinite(in.y_max);
}

bool KruglovaA2DMuitSEQ::PreProcessingImpl() {
  GetOutput() = {0.0, 0.0, std::numeric_limits<double>::max()};
  return true;
}

bool KruglovaA2DMuitSEQ::RunImpl() {
  const auto &in = GetInput();
  const double r_param = 2.0;

  auto compute_z = [&](double x_val, double &best_y) {
    return Solve1DStrongin([&](double y_val) { return ObjectiveFunction(x_val, y_val); }, in.y_min, in.y_max, in.eps,
                           std::max(50, in.max_iters / 10), best_y);
  };

  std::vector<Trial2D> x_trials;
  const int init_points = 20;

  for (int i = 0; i < init_points; ++i) {
    double x_val = in.x_min + (in.x_max - in.x_min) * static_cast<double>(i) / static_cast<double>(init_points - 1);

    double y_res = 0.0;
    double z_res = compute_z(x_val, y_res);

    x_trials.push_back({x_val, y_res, z_res});
  }

  for (size_t i = 0; i < x_trials.size(); ++i) {
    size_t min_idx = i;
    for (size_t j = i + 1; j < x_trials.size(); ++j) {
      if (x_trials[j].x < x_trials[min_idx].x) {
        min_idx = j;
      }
    }
    std::swap(x_trials[i], x_trials[min_idx]);
  }

  for (int iter = 0; iter < in.max_iters; ++iter) {
    double m_val = CalculateM(x_trials);
    double m_scaled = (m_val > 0.0) ? (r_param * m_val) : 1.0;

    size_t best_idx = FindBestInterval(x_trials, m_scaled);

    if ((x_trials[best_idx + 1].x - x_trials[best_idx].x) < in.eps) {
      break;
    }

    double x_new = 0.5 * (x_trials[best_idx + 1].x + x_trials[best_idx].x) -
                   (x_trials[best_idx + 1].z - x_trials[best_idx].z) / (2.0 * m_scaled);

    double y_new = 0.0;
    double z_new = compute_z(x_new, y_new);

    Trial2D new_trial{x_new, y_new, z_new};

    size_t pos = 0;
    while (pos < x_trials.size() && x_trials[pos].x < x_new) {
      ++pos;
    }

    if (pos == x_trials.size() || std::abs(x_trials[pos].x - x_new) > 1e-12) {
      x_trials.insert(x_trials.begin() + static_cast<std::ptrdiff_t>(pos), new_trial);
    }
  }

  size_t best_idx = 0;
  for (size_t i = 1; i < x_trials.size(); ++i) {
    if (x_trials[i].z < x_trials[best_idx].z) {
      best_idx = i;
    }
  }

  GetOutput() = {x_trials[best_idx].x, x_trials[best_idx].y, x_trials[best_idx].z};
  return true;
}

bool KruglovaA2DMuitSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
