#include "krykov_e_multistep_sad/seq/include/ops_seq.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "krykov_e_multistep_sad/common/include/common.hpp"

namespace krykov_e_multistep_sad {

namespace {

constexpr double kEps = 1e-4;
constexpr int kMaxIter = 1000;

double EvaluateCenter(const Function2D &f, Region &r) {
  const double xc = 0.5 * (r.x_min + r.x_max);
  const double yc = 0.5 * (r.y_min + r.y_max);
  r.value = f(xc, yc);
  return r.value;
}

}  // namespace

KrykovEMultistepSADSEQ::KrykovEMultistepSADSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovEMultistepSADSEQ::ValidationImpl() {
  const auto &[f, x1, x2, y1, y2] = GetInput();
  return static_cast<bool>(f) && x1 < x2 && y1 < y2;
}

bool KrykovEMultistepSADSEQ::PreProcessingImpl() {
  return true;
}

bool KrykovEMultistepSADSEQ::RunImpl() {
  const auto &[f, x_min, x_max, y_min, y_max] = GetInput();

  std::vector<Region> regions;
  regions.push_back({x_min, x_max, y_min, y_max, 0.0});
  EvaluateCenter(f, regions.front());

  for (int iter = 0; iter < kMaxIter; ++iter) {
    auto best_it = std::min_element(regions.begin(), regions.end(),
                                    [](const Region &a, const Region &b) { return a.value < b.value; });

    Region best = *best_it;
    regions.erase(best_it);

    const double dx = best.x_max - best.x_min;
    const double dy = best.y_max - best.y_min;

    if (std::max(dx, dy) < kEps) {
      break;
    }

    if (dx >= dy) {
      double xm = 0.5 * (best.x_min + best.x_max);
      Region r1{best.x_min, xm, best.y_min, best.y_max, 0.0};
      Region r2{xm, best.x_max, best.y_min, best.y_max, 0.0};
      EvaluateCenter(f, r1);
      EvaluateCenter(f, r2);
      regions.push_back(r1);
      regions.push_back(r2);
    } else {
      double ym = 0.5 * (best.y_min + best.y_max);
      Region r1{best.x_min, best.x_max, best.y_min, ym, 0.0};
      Region r2{best.x_min, best.x_max, ym, best.y_max, 0.0};
      EvaluateCenter(f, r1);
      EvaluateCenter(f, r2);
      regions.push_back(r1);
      regions.push_back(r2);
    }
  }

  const auto &best = *std::min_element(regions.begin(), regions.end(),
                                       [](const Region &a, const Region &b) { return a.value < b.value; });

  const double x = 0.5 * (best.x_min + best.x_max);
  const double y = 0.5 * (best.y_min + best.y_max);

  GetOutput() = {x, y, best.value};
  return true;
}

bool KrykovEMultistepSADSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_multistep_sad
