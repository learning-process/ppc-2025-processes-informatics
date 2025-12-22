#include "levonychev_i_multistep_2d_optimization/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "levonychev_i_multistep_2d_optimization/common/include/optimization_common.hpp"

namespace levonychev_i_multistep_2d_optimization {

LevonychevIMultistep2dOptimizationSEQ::LevonychevIMultistep2dOptimizationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OptimizationResult();
}

bool LevonychevIMultistep2dOptimizationSEQ::ValidationImpl() {
  const auto &params = GetInput();

  bool is_correct_ranges = (params.x_min < params.x_max) && (params.y_min < params.y_max);
  bool isnt_correct_func = !params.func;
  bool is_correct_params = (params.num_steps > 0) && (params.grid_size_step1 > 0) && (params.candidates_per_step > 0);

  return is_correct_ranges && !isnt_correct_func && is_correct_params;
}

bool LevonychevIMultistep2dOptimizationSEQ::PreProcessingImpl() {
  GetOutput() = OptimizationResult();
  return true;
}

std::vector<Point> LevonychevIMultistep2dOptimizationSEQ::GenerateGridPoints(double x_min, double x_max, double y_min,
                                                                             double y_max, int grid_size) {
  const auto &params = GetInput();
  std::vector<Point> grid_points;

  double step_x = (x_max - x_min) / (grid_size - 1);
  double step_y = (y_max - y_min) / (grid_size - 1);

  for (int i = 0; i < grid_size; ++i) {
    double x = x_min + (i * step_x);
    for (int j = 0; j < grid_size; ++j) {
      double y = y_min + (j * step_y);
      x = std::max(params.x_min, std::min(params.x_max, x));
      y = std::max(params.y_min, std::min(params.y_max, y));

      double value = params.func(x, y);
      grid_points.emplace_back(x, y, value);
    }
  }

  return grid_points;
}

std::vector<Point> LevonychevIMultistep2dOptimizationSEQ::SelectTopCandidates(const std::vector<Point> &points,
                                                                              int num_candidates) const {
  std::vector<Point> sorted_points = points;
  std::ranges::sort(sorted_points, [](const Point &a, const Point &b) { return a.value < b.value; });

  int num_to_select = std::min(num_candidates, static_cast<int>(sorted_points.size()));
  return std::vector<Point>(sorted_points.begin(), sorted_points.begin() + num_to_select);
}

void LevonychevIMultistep2dOptimizationSEQ::UpdateSearchRegionFromCandidates(const std::vector<Point> &candidates,
                                                                             double &x_min, double &x_max,
                                                                             double &y_min, double &y_max) {
  const auto &params = GetInput();

  if (candidates.empty()) {
    return;
  }

  double min_x = candidates[0].x;
  double max_x = candidates[0].x;
  double min_y = candidates[0].y;
  double max_y = candidates[0].y;

  for (const auto &cand : candidates) {
    min_x = std::min(min_x, cand.x);
    max_x = std::max(max_x, cand.x);
    min_y = std::min(min_y, cand.y);
    max_y = std::max(max_y, cand.y);
  }

  double margin_x = (max_x - min_x) * 0.1;
  double margin_y = (max_y - min_y) * 0.1;

  x_min = std::max(params.x_min, min_x - margin_x);
  x_max = std::min(params.x_max, max_x + margin_x);
  y_min = std::max(params.y_min, min_y - margin_y);
  y_max = std::min(params.y_max, max_y + margin_y);

  if (x_min >= x_max) {
    x_min = params.x_min;
    x_max = params.x_max;
  }
  if (y_min >= y_max) {
    y_min = params.y_min;
    y_max = params.y_max;
  }
}

Point LevonychevIMultistep2dOptimizationSEQ::ApplyLocalOptimizationToCandidates(const std::vector<Point> &candidates) {
  const auto &params = GetInput();
  Point best_point = candidates[0];

  for (const auto &cand : candidates) {
    Point optimized =
        LocalOptimization(params.func, cand.x, cand.y, params.x_min, params.x_max, params.y_min, params.y_max);

    if (optimized.value < best_point.value) {
      best_point = optimized;
    }
  }

  return best_point;
}

void LevonychevIMultistep2dOptimizationSEQ::SetFinalResult(const std::vector<Point> &candidates) {
  const auto &params = GetInput();
  auto &result = GetOutput();

  if (params.use_local_optimization && !candidates.empty()) {
    Point best_point = ApplyLocalOptimizationToCandidates(candidates);
    result.x_min = best_point.x;
    result.y_min = best_point.y;
    result.value = best_point.value;
  } else if (!candidates.empty()) {
    result.x_min = candidates[0].x;
    result.y_min = candidates[0].y;
    result.value = candidates[0].value;
  } else {
    result.x_min = (params.x_min + params.x_max) / 2.0;
    result.y_min = (params.y_min + params.y_max) / 2.0;
    result.value = params.func(result.x_min, result.y_min);
  }
}

bool LevonychevIMultistep2dOptimizationSEQ::RunImpl() {
  const auto &params = GetInput();
  auto &result = GetOutput();

  double current_x_min = params.x_min;
  double current_x_max = params.x_max;
  double current_y_min = params.y_min;
  double current_y_max = params.y_max;

  std::vector<Point> candidates;

  for (int step = 0; step < params.num_steps; ++step) {
    int grid_size = params.grid_size_step1 * (1 << step);

    std::vector<Point> grid_points =
        GenerateGridPoints(current_x_min, current_x_max, current_y_min, current_y_max, grid_size);
    candidates = SelectTopCandidates(grid_points, params.candidates_per_step);

    if (step < params.num_steps - 1 && !candidates.empty()) {
      UpdateSearchRegionFromCandidates(candidates, current_x_min, current_x_max, current_y_min, current_y_max);
    }

    result.iterations += grid_size * grid_size;
  }

  SetFinalResult(candidates);
  return true;
}

bool LevonychevIMultistep2dOptimizationSEQ::PostProcessingImpl() {
  const auto &params = GetInput();
  auto &result = GetOutput();

  return !(result.x_min < params.x_min || result.x_min > params.x_max || result.y_min < params.y_min ||
           result.y_min > params.y_max);
}

}  // namespace levonychev_i_multistep_2d_optimization
