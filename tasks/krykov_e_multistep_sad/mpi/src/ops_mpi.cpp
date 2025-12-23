#include "krykov_e_multistep_sad/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>  // для std::ranges::copy, std::ranges::remove_if
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

int ComputeStartIndex(int rank, int regions_per_proc, int remainder) {
  return rank * regions_per_proc + (rank < remainder ? rank : remainder);
}

int ComputeEndIndex(int start_idx, int regions_per_proc, int rank, int remainder) {
  return start_idx + regions_per_proc + (rank < remainder ? 1 : 0);
}

Region SplitRegionX(const Region &r, double xm) {
  return Region{.x_min = r.x_min, .x_max = xm, .y_min = r.y_min, .y_max = r.y_max, .value = 0.0};
}

Region SplitRegionXRight(const Region &r, double xm) {
  return Region{.x_min = xm, .x_max = r.x_max, .y_min = r.y_min, .y_max = r.y_max, .value = 0.0};
}

Region SplitRegionY(const Region &r, double ym) {
  return Region{.x_min = r.x_min, .x_max = r.x_max, .y_min = r.y_min, .y_max = ym, .value = 0.0};
}

Region SplitRegionYTop(const Region &r, double ym) {
  return Region{.x_min = r.x_min, .x_max = r.x_max, .y_min = ym, .y_max = r.y_max, .value = 0.0};
}

// Вычисление локального лучшего региона
Region FindLocalBest(const Function2D &f, std::vector<Region> &local_regions, int start_idx, int end_idx) {
  Region best{.x_min = 0, .x_max = 0, .y_min = 0, .y_max = 0, .value = std::numeric_limits<double>::max()};
  bool initialized = false;
  for (int i = start_idx; i < end_idx; ++i) {
    EvaluateCenter(f, local_regions[i]);
    if (!initialized || local_regions[i].value < best.value) {
      best = local_regions[i];
      initialized = true;
    }
  }
  if (!initialized) {
    best.value = 9999;
  }
  return best;
}

// Обновление списка регионов после разбиения
void UpdateRegions(std::vector<Region> &regions, const Region &global_best) {
  double dx = global_best.x_max - global_best.x_min;
  double dy = global_best.y_max - global_best.y_min;

  auto sub = std::ranges::remove_if(regions, [&](const Region &r) {
    return r.x_min == global_best.x_min && r.x_max == global_best.x_max && r.y_min == global_best.y_min &&
           r.y_max == global_best.y_max;
  });
  regions.erase(sub.begin(), sub.end());

  if (dx >= dy) {
    double xm = 0.5 * (global_best.x_min + global_best.x_max);
    regions.push_back(SplitRegionX(global_best, xm));
    regions.push_back(SplitRegionXRight(global_best, xm));
  } else {
    double ym = 0.5 * (global_best.y_min + global_best.y_max);
    regions.push_back(SplitRegionY(global_best, ym));
    regions.push_back(SplitRegionYTop(global_best, ym));
  }
}

}  // namespace

KrykovEMultistepSADMPI::KrykovEMultistepSADMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovEMultistepSADMPI::ValidationImpl() {
  const auto &[f, x1, x2, y1, y2] = GetInput();
  return static_cast<bool>(f) && x1 < x2 && y1 < y2;
}

bool KrykovEMultistepSADMPI::PreProcessingImpl() {
  return true;
}

bool KrykovEMultistepSADMPI::RunImpl() {
  int size{}, rank{};
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &[f, x_min, x_max, y_min, y_max] = GetInput();

  std::vector<Region> regions;
  if (rank == 0) {
    regions.push_back(Region{.x_min = x_min, .x_max = x_max, .y_min = y_min, .y_max = y_max, .value = 0.0});
    EvaluateCenter(f, regions.front());
  }

  int stop_flag = 0;
  for (int iter = 0; iter < kMaxIter && stop_flag == 0; ++iter) {
    int n_regions = 0;
    if (rank == 0) {
      n_regions = static_cast<int>(regions.size());
    }

    MPI_Bcast(&n_regions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n_regions == 0) {
      stop_flag = 1;
      MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
      break;
    }

    std::vector<Region> local_regions(n_regions);
    if (rank == 0) {
      std::ranges::copy(regions, local_regions.begin());
    }
    MPI_Bcast(local_regions.data(), static_cast<int>(n_regions * sizeof(Region)), MPI_BYTE, 0, MPI_COMM_WORLD);

    int regions_per_proc = n_regions / size;
    int remainder = n_regions % size;
    int start_idx = ComputeStartIndex(rank, regions_per_proc, remainder);
    int end_idx = ComputeEndIndex(start_idx, regions_per_proc, rank, remainder);

    Region local_best = FindLocalBest(f, local_regions, start_idx, end_idx);

    struct {
      double value;
      int rank;
    } local_val{.value = local_best.value, .rank = rank}, global_val{.value = 0.0, .rank = 0};

    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    Region global_best = local_best;
    if (rank == global_val.rank) {
      global_best = local_best;
    }

    MPI_Bcast(&global_best, sizeof(Region), MPI_BYTE, global_val.rank, MPI_COMM_WORLD);

    if (rank == 0) {
      if (std::max(global_best.x_max - global_best.x_min, global_best.y_max - global_best.y_min) < kEps) {
        stop_flag = 1;
      } else {
        UpdateRegions(regions, global_best);
      }
    }

    MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  Region best_region{.x_min = 0, .x_max = 0, .y_min = 0, .y_max = 0, .value = std::numeric_limits<double>::max()};
  double x = 0.0, y = 0.0, value = 0.0;

  if (rank == 0) {
    for (auto &r : regions) {
      EvaluateCenter(f, r);
    }
    best_region = *std::ranges::min_element(regions, {}, &Region::value);

    x = 0.5 * (best_region.x_min + best_region.x_max);
    y = 0.5 * (best_region.y_min + best_region.y_max);
    value = best_region.value;
  }

  MPI_Bcast(&x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = {x, y, value};
  return true;
}

bool KrykovEMultistepSADMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_multistep_sad
