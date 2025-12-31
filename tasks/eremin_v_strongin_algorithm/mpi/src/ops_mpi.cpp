#include "eremin_v_strongin_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <list>
#include <tuple>

#include "eremin_v_strongin_algorithm/common/include/common.hpp"

namespace eremin_v_strongin_algorithm {

EreminVStronginAlgorithmMPI::EreminVStronginAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool EreminVStronginAlgorithmMPI::ValidationImpl() {
  auto &input = GetInput();
  return (std::get<0>(input) < std::get<1>(input)) && (std::get<2>(input) > 0) && (std::get<2>(input) <= 100000000) &&
         (std::get<0>(input) >= -1e9) && (std::get<0>(input) <= 1e9) && (std::get<1>(input) >= -1e9) &&
         (std::get<1>(input) <= 1e9) && (GetOutput() == 0);
}

bool EreminVStronginAlgorithmMPI::PreProcessingImpl() {
  return true;
}

bool EreminVStronginAlgorithmMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &input = GetInput();
  double lower_bound = 0.0;
  double upper_bound = 0.0;
  int steps = 0;

  if (rank == 0) {
    lower_bound = std::get<0>(input);
    upper_bound = std::get<1>(input);
    steps = std::get<2>(input);
  }

  MPI_Bcast(&lower_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&upper_bound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto function = std::get<3>(input);

  std::list<double> points_list = {lower_bound, upper_bound};
  std::list<double> values_list = {function(lower_bound), function(upper_bound)};

  const double r_constant = 2.0;

  for (int step = 0; step < steps; step++) {
    int num_intervals = points_list.size() - 1;

    double local_max_slope = 0.0;
    auto it_x_left = points_list.begin();
    auto it_y_left = values_list.begin();

    for (int i = 0; i < num_intervals; ++i) {
      auto it_x_right = std::next(it_x_left);
      auto it_y_right = std::next(it_y_left);

      if (i % size == rank) {
        double slope = std::abs(*it_y_right - *it_y_left) / (*it_x_right - *it_x_left);
        if (slope > local_max_slope) {
          local_max_slope = slope;
        }
      }
      it_x_left++;
      it_y_left++;
    }

    double global_max_slope;
    MPI_Allreduce(&local_max_slope, &global_max_slope, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double M = (global_max_slope > 0) ? r_constant * global_max_slope : 1.0;

    double local_max_char = -1e18;
    int local_best_idx = -1;

    it_x_left = points_list.begin();
    it_y_left = values_list.begin();

    for (int i = 0; i < num_intervals; ++i) {
      auto it_x_right = std::next(it_x_left);
      auto it_y_right = std::next(it_y_left);

      if (i % size == rank) {
        double dx = *it_x_right - *it_x_left;
        double dy = *it_y_right - *it_y_left;
        double ch = M * dx + (dy * dy) / (M * dx) - 2.0 * (*it_y_right + *it_y_left);
        if (ch > local_max_char) {
          local_max_char = ch;
          local_best_idx = i;
        }
      }
      it_x_left++;
      it_y_left++;
    }

    struct {
      double val;
      int rank;
    } local_res{local_max_char, rank}, global_res;
    MPI_Allreduce(&local_res, &global_res, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    int best_idx = -1;
    if (rank == global_res.rank) {
      best_idx = local_best_idx;
    }
    MPI_Bcast(&best_idx, 1, MPI_INT, global_res.rank, MPI_COMM_WORLD);

    auto it_x_ins = std::next(points_list.begin(), best_idx);
    auto it_y_ins = std::next(values_list.begin(), best_idx);

    auto it_x_next = std::next(it_x_ins);
    auto it_y_next = std::next(it_y_ins);

    double x_new = 0.5 * (*it_x_next + *it_x_ins) - (*it_y_next - *it_y_ins) / (2.0 * M);
    double y_new = function(x_new);

    points_list.insert(it_x_next, x_new);
    values_list.insert(it_y_next, y_new);
  }

  auto min_it = std::min_element(values_list.begin(), values_list.end());
  auto dist = std::distance(values_list.begin(), min_it);
  GetOutput() = *std::next(points_list.begin(), dist);

  return true;
}

bool EreminVStronginAlgorithmMPI::PostProcessingImpl() {
  return true;
}

}  // namespace eremin_v_strongin_algorithm
