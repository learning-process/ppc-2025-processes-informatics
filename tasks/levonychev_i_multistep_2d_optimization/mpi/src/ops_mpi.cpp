#include "levonychev_i_multistep_2d_optimization/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "levonychev_i_multistep_2d_optimization/common/include/optimization_common.hpp"

namespace levonychev_i_multistep_2d_optimization {

LevonychevIMultistep2dOptimizationMPI::LevonychevIMultistep2dOptimizationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OptimizationResult();
}

bool LevonychevIMultistep2dOptimizationMPI::ValidationImpl() {
  const auto &params = GetInput();

  bool is_correct_ranges = (params.x_min < params.x_max) && (params.y_min < params.y_max);
  bool isnt_correct_func = !params.func;
  bool is_correct_params = (params.num_steps > 0) && (params.grid_size_step1 > 0) && (params.candidates_per_step > 0);

  return is_correct_ranges && !isnt_correct_func && is_correct_params;
}

bool LevonychevIMultistep2dOptimizationMPI::PreProcessingImpl() {
  GetOutput() = OptimizationResult();
  return true;
}

void LevonychevIMultistep2dOptimizationMPI::InitializeRegions(int rank, int size, const OptimizationParams &params,
                                                              SearchRegion &my_region) {
  std::vector<SearchRegion> initial_regions;
  if (rank == 0) {
    double total_width = params.x_max - params.x_min;
    double width_per_process = total_width / size;

    initial_regions.resize(size);
    for (int proc = 0; proc < size; ++proc) {
      double x_min_proc = params.x_min + (proc * width_per_process);
      double x_max_proc = (proc == size - 1) ? params.x_max : params.x_min + ((proc + 1) * width_per_process);
      initial_regions[proc] = SearchRegion(x_min_proc, x_max_proc, params.y_min, params.y_max);
    }
  }
  MPI_Scatter(initial_regions.data(), sizeof(SearchRegion), MPI_BYTE, &my_region, sizeof(SearchRegion), MPI_BYTE, 0,
              MPI_COMM_WORLD);
}

void LevonychevIMultistep2dOptimizationMPI::GatherAndSelectCandidates(int rank,
                                                                      const std::vector<Point> &local_candidates,
                                                                      std::vector<Point> &all_candidates) {
  std::vector<Point> recv_buffer;
  if (rank == 0) {
    recv_buffer.resize(all_candidates.size());
  }

  MPI_Gather(local_candidates.data(), static_cast<int>(local_candidates.size() * sizeof(Point)),
             MPI_BYTE, recv_buffer.data(), static_cast<int>(local_candidates.size() * sizeof(Point)), MPI_BYTE, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<Point> valid_candidates;
    for (const auto &point : recv_buffer) {
      if (point.value < std::numeric_limits<double>::max()) {
        valid_candidates.push_back(point);
      }
    }
    std::ranges::sort(valid_candidates, [](const Point &a, const Point &b) { return a.value < b.value; });

    int num_global = std::min(static_cast<int>(local_candidates.size()), static_cast<int>(valid_candidates.size()));
    all_candidates.assign(valid_candidates.begin(), valid_candidates.begin() + num_global);
  }
}

void LevonychevIMultistep2dOptimizationMPI::BuildNewRegions(const OptimizationParams &params,
                                                            const std::vector<Point> &all_candidates, int step,
                                                            std::vector<SearchRegion> &new_regions) {
  if (!all_candidates.empty()) {
    for (const auto &cand : all_candidates) {
      double margin_x = (params.x_max - params.x_min) * 0.05 / (1 << step);
      double margin_y = (params.y_max - params.y_min) * 0.05 / (1 << step);

      SearchRegion new_region(std::max(params.x_min, cand.x - margin_x), std::min(params.x_max, cand.x + margin_x),
                              std::max(params.y_min, cand.y - margin_y), std::min(params.y_max, cand.y + margin_y));
      new_regions.push_back(new_region);
    }
  } else {
    new_regions.emplace_back(params.x_min, params.x_max, params.y_min, params.y_max);
  }
}

void LevonychevIMultistep2dOptimizationMPI::DistributeRegionsToProcesses(
    int rank, int size, const OptimizationParams &params, const std::vector<SearchRegion> &new_regions,
    SearchRegion &my_region) {
  if (rank == 0) {
    std::vector<SearchRegion> regions_to_scatter(size);

    if (new_regions.empty()) {
      for (int proc = 0; proc < size; ++proc) {
        regions_to_scatter[proc] = SearchRegion(params.x_min, params.x_max, params.y_min, params.y_max);
      }
    } else if (std::cmp_greater_equal(new_regions.size(), size)) {
      for (int proc = 0; proc < size; ++proc) {
        regions_to_scatter[proc] = new_regions[proc];
      }
    } else {
      int regions_count = static_cast<int>(new_regions.size());
      for (int proc = 0; proc < size; ++proc) {
        int region_idx = proc % regions_count;
        const auto &base_region = new_regions[region_idx];
        int processes_per_region = (size + regions_count - 1) / regions_count;
        int local_idx = proc / regions_count;

        double region_width = base_region.x_max - base_region.x_min;
        double width_per_process = region_width / processes_per_region;
        double x_min_proc = base_region.x_min + (local_idx * width_per_process);
        double x_max_proc = (local_idx == processes_per_region - 1) ? base_region.x_max
                                                                    : base_region.x_min + ((local_idx + 1) * width_per_process);

        regions_to_scatter[proc] = SearchRegion(x_min_proc, x_max_proc, base_region.y_min, base_region.y_max);
      }
    }

    MPI_Scatter(regions_to_scatter.data(), sizeof(SearchRegion), MPI_BYTE, &my_region, sizeof(SearchRegion), MPI_BYTE, 0,
                MPI_COMM_WORLD);
  } else {
    MPI_Scatter(nullptr, sizeof(SearchRegion), MPI_BYTE, &my_region, sizeof(SearchRegion), MPI_BYTE, 0, MPI_COMM_WORLD);
  }

  if (my_region.x_min >= my_region.x_max || my_region.y_min >= my_region.y_max) {
    my_region = SearchRegion(params.x_min, params.x_max, params.y_min, params.y_max);
  }
}

void LevonychevIMultistep2dOptimizationMPI::ScatterNewRegions(int rank, int size, const OptimizationParams &params,
                                                              const std::vector<Point> &all_candidates, int step,
                                                              SearchRegion &my_region) {
  std::vector<SearchRegion> new_regions;
  if (rank == 0) {
    BuildNewRegions(params, all_candidates, step, new_regions);
  }
  DistributeRegionsToProcesses(rank, size, params, new_regions, my_region);
}

void LevonychevIMultistep2dOptimizationMPI::ProcessLocalCandidates(const OptimizationParams &params,
                                                                   const std::vector<Point> &local_points,
                                                                   std::vector<Point> &local_candidates) {
  int num_local_candidates = std::min(params.candidates_per_step, static_cast<int>(local_points.size()));
  local_candidates.assign(local_points.begin(), local_points.begin() + num_local_candidates);

  while (std::cmp_less(local_candidates.size(), params.candidates_per_step)) {
    local_candidates.emplace_back();
  }
}

void LevonychevIMultistep2dOptimizationMPI::ExecuteOptimizationSteps(int rank, int size,
                                                                     const OptimizationParams &params,
                                                                     SearchRegion &my_region,
                                                                     OptimizationResult &result) {
  for (int step = 0; step < params.num_steps; ++step) {
    int grid_size = params.grid_size_step1 * (1 << step);
    std::vector<Point> local_points = SearchInRegion(params.func, my_region, grid_size);

    std::vector<Point> local_candidates;
    ProcessLocalCandidates(params, local_points, local_candidates);

    std::vector<Point> all_candidates;
    if (rank == 0) {
      all_candidates.resize(static_cast<size_t>(params.candidates_per_step) * static_cast<size_t>(size));
    }

    MPI_Gather(local_candidates.data(), static_cast<int>(params.candidates_per_step * sizeof(Point)), MPI_BYTE,
               (rank == 0) ? all_candidates.data() : nullptr,
               static_cast<int>(params.candidates_per_step * sizeof(Point)), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      std::vector<Point> valid_candidates;
      for (const auto &point : all_candidates) {
        if (point.value < std::numeric_limits<double>::max()) {
          valid_candidates.push_back(point);
        }
      }

      std::ranges::sort(valid_candidates, [](const Point &a, const Point &b) { return a.value < b.value; });

      int num_global_candidates = std::min(params.candidates_per_step, static_cast<int>(valid_candidates.size()));
      all_candidates.assign(valid_candidates.begin(), valid_candidates.begin() + num_global_candidates);
    }

    if (step < params.num_steps - 1) {
      ScatterNewRegions(rank, size, params, all_candidates, step, my_region);
    }

    result.iterations += grid_size * grid_size;
  }
}

void LevonychevIMultistep2dOptimizationMPI::GatherFinalPoints(int rank, int size, const Point &my_best,
                                                              std::vector<Point> &all_final_points) {
  if (rank == 0) {
    all_final_points.resize(static_cast<size_t>(size));
  }

  MPI_Gather(&const_cast<Point &>(my_best), sizeof(Point), MPI_BYTE, (rank == 0) ? all_final_points.data() : nullptr,
             sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
}

Point LevonychevIMultistep2dOptimizationMPI::SelectGlobalBest(int rank, const OptimizationParams &params,
                                                             const std::vector<Point> &all_final_points) {
  Point global_best;
  if (rank == 0) {
    std::vector<Point> valid_points;
    for (const auto &point : all_final_points) {
      if (point.value < std::numeric_limits<double>::max()) {
        valid_points.push_back(point);
      }
    }

    if (!valid_points.empty()) {
      std::ranges::sort(valid_points, [](const Point &a, const Point &b) { return a.value < b.value; });
      global_best = valid_points[0];
    } else {
      global_best = Point((params.x_min + params.x_max) / 2.0, (params.y_min + params.y_max) / 2.0,
                          params.func((params.x_min + params.x_max) / 2.0, (params.y_min + params.y_max) / 2.0));
    }
  }
  return global_best;
}

Point LevonychevIMultistep2dOptimizationMPI::FindGlobalBest(int rank, int size, const OptimizationParams &params,
                                                            const SearchRegion &my_region) {
  Point my_best;

  if (my_region.x_min < my_region.x_max && my_region.y_min < my_region.y_max) {
    int power_of_2 = 1;
    for (int i = 0; i < params.num_steps - 1; ++i) {
      power_of_2 *= 2;
    }
    std::vector<Point> local_points = SearchInRegion(params.func, my_region, params.grid_size_step1 * power_of_2);

    if (!local_points.empty()) {
      my_best = local_points[0];
      if (params.use_local_optimization) {
        my_best = LocalOptimization(params.func, my_best.x, my_best.y, params.x_min, params.x_max, params.y_min,
                                    params.y_max);
      }
    }
  }

  std::vector<Point> all_final_points;
  GatherFinalPoints(rank, size, my_best, all_final_points);

  Point global_best = SelectGlobalBest(rank, params, all_final_points);

  MPI_Bcast(&global_best, sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

  return global_best;
}

bool LevonychevIMultistep2dOptimizationMPI::RunImpl() {
  const auto &params = GetInput();
  auto &result = GetOutput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  SearchRegion my_region;
  InitializeRegions(rank, size, params, my_region);

  ExecuteOptimizationSteps(rank, size, params, my_region, result);

  Point global_best = FindGlobalBest(rank, size, params, my_region);

  result.x_min = global_best.x;
  result.y_min = global_best.y;
  result.value = global_best.value;

  int local_iterations = result.iterations;
  int global_iterations = 0;
  MPI_Allreduce(&local_iterations, &global_iterations, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  result.iterations = global_iterations;

  return true;
}

bool LevonychevIMultistep2dOptimizationMPI::PostProcessingImpl() {
  const auto &params = GetInput();
  auto &result = GetOutput();

  return result.x_min >= params.x_min && result.x_min <= params.x_max && result.y_min >= params.y_min &&
           result.y_min <= params.y_max;
}

}  // namespace levonychev_i_multistep_2d_optimization
