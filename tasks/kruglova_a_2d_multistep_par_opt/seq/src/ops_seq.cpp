#include "kruglova_a_2d_multistep_par_opt/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "kruglova_a_2d_multistep_par_opt/common/include/common.hpp"

namespace kruglova_a_2d_multistep_par_opt {

struct Trial1D {
  double x;
  double z;
};

struct Trial2D {
  double x;
  double y;
  double z;
};

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

double KruglovaA2DMuitSEQ::Solve1D(std::function<double(double)> func, double a, double b, double eps, int max_iters,
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

bool KruglovaA2DMuitSEQ::RunImpl() {
  const auto &in = GetInput();
  const double r = 2.0;

  auto compute_Z = [&](double x, double &best_y) {
    return Solve1D([&](double y) { return ObjectiveFunction(x, y); }, in.y_min, in.y_max, in.eps,
                   std::max(50, in.max_iters / 10), best_y);
  };

  std::vector<Trial2D> x_trials;
  const int init_points = 40;

  for (int i = 0; i < init_points; ++i) {
    double x = in.x_min + (in.x_max - in.x_min) * i / (double)(init_points - 1);
    double y_res = 0.0;
    double z_res = compute_Z(x, y_res);
    x_trials.push_back({x, y_res, z_res});
  }

  std::sort(x_trials.begin(), x_trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });

  for (int iter = 0; iter < in.max_iters; ++iter) {
    double M = 0.0;
    for (size_t i = 0; i + 1 < x_trials.size(); ++i) {
      double dx = x_trials[i + 1].x - x_trials[i].x;
      if (dx > 1e-15) {
        M = std::max(M, std::abs(x_trials[i + 1].z - x_trials[i].z) / dx);
      }
    }

    double m = (M > 0.0) ? r * M : 1.0;

    double max_R = -std::numeric_limits<double>::infinity();
    size_t best_idx = 0;

    for (size_t i = 0; i + 1 < x_trials.size(); ++i) {
      double dx = x_trials[i + 1].x - x_trials[i].x;
      double dz = x_trials[i + 1].z - x_trials[i].z;
      double R = m * dx + (dz * dz) / (m * dx) - 2.0 * (x_trials[i + 1].z + x_trials[i].z);

      if (R > max_R) {
        max_R = R;
        best_idx = i;
      }
    }

    if (x_trials[best_idx + 1].x - x_trials[best_idx].x < in.eps) {
      break;
    }

    double x_new = 0.5 * (x_trials[best_idx + 1].x + x_trials[best_idx].x) -
                   (x_trials[best_idx + 1].z - x_trials[best_idx].z) / (2.0 * m);

    double y_new = 0.0;
    double z_new = compute_Z(x_new, y_new);
    Trial2D new_trial = {x_new, y_new, z_new};

    auto it = std::lower_bound(x_trials.begin(), x_trials.end(), new_trial,
                               [](const Trial2D &a, const Trial2D &b) { return a.x < b.x; });

    if (it == x_trials.end() || std::abs(it->x - x_new) > 1e-12) {
      x_trials.insert(it, new_trial);
    }
  }

  auto best =
      std::min_element(x_trials.begin(), x_trials.end(), [](const Trial2D &a, const Trial2D &b) { return a.z < b.z; });

  GetOutput() = {best->x, best->y, best->z};
  return true;
}

bool KruglovaA2DMuitSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
