#include "krykov_e_multistep_sad/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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
  int size = 0, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &[f, x_min, x_max, y_min, y_max] = GetInput();

  std::vector<Region> regions;
  if (rank == 0) {
    regions.push_back({x_min, x_max, y_min, y_max, 0.0});
    EvaluateCenter(f, regions.front());
  }

  int stop_flag = 0;

  for (int iter = 0; iter < kMaxIter && !stop_flag; ++iter) {
    int n_regions = regions.size();

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
      std::copy(regions.begin(), regions.end(), local_regions.begin());
    }
    MPI_Bcast(local_regions.data(), n_regions * sizeof(Region), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Каждый процесс вычисляет локальный лучший
    int regions_per_proc = n_regions / size;
    int remainder = n_regions % size;
    int start_idx = rank * regions_per_proc + std::min(rank, remainder);
    int end_idx = start_idx + regions_per_proc + (rank < remainder ? 1 : 0);

    Region local_best = {0, 0, 0, 0, std::numeric_limits<double>::max()};
    bool local_best_initialized = false;

    // Вычисляем центры и находим локальный минимум
    for (int i = start_idx; i < end_idx; ++i) {
      EvaluateCenter(f, local_regions[i]);
      if (!local_best_initialized || local_regions[i].value < local_best.value) {
        local_best = local_regions[i];
        local_best_initialized = true;
      }
    }

    if (!local_best_initialized) {
      local_best.value = 9999;
    }

    // Master получает глобально лучший регион
    struct {
      double value;
      int rank;
    } local_val{local_best.value, rank}, global_val;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    Region global_best = local_best;  // Инициализация
    if (rank == global_val.rank) {
      global_best = local_best;
    }
    MPI_Bcast(&global_best, sizeof(Region), MPI_BYTE, global_val.rank, MPI_COMM_WORLD);

    if (rank == 0) {
      double dx = global_best.x_max - global_best.x_min;
      double dy = global_best.y_max - global_best.y_min;
      if (std::max(dx, dy) < kEps) {
        stop_flag = 1;
      } else {
        regions.erase(std::remove_if(regions.begin(), regions.end(),
                                     [&](const Region &r) {
          return r.x_min == global_best.x_min && r.x_max == global_best.x_max && r.y_min == global_best.y_min &&
                 r.y_max == global_best.y_max;
        }),
                      regions.end());
        if (dx >= dy) {
          double xm = 0.5 * (global_best.x_min + global_best.x_max);
          regions.push_back({global_best.x_min, xm, global_best.y_min, global_best.y_max, 0.0});
          regions.push_back({xm, global_best.x_max, global_best.y_min, global_best.y_max, 0.0});
        } else {
          double ym = 0.5 * (global_best.y_min + global_best.y_max);
          regions.push_back({global_best.x_min, global_best.x_max, global_best.y_min, ym, 0.0});
          regions.push_back({global_best.x_min, global_best.x_max, ym, global_best.y_max, 0.0});
        }
      }
    }

    MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  double x = 0.0;
  double y = 0.0;
  double value = 0.0;

  Region best_region;

  if (rank == 0) {
    for (auto &r : regions) {
      EvaluateCenter(f, r);
    }

    best_region = regions.front();
    for (const auto &r : regions) {
      if (r.value < best_region.value) {
        best_region = r;
      }
    }

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
