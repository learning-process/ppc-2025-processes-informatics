#include "fatehov_k_matrix_max_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"

namespace fatehov_k_matrix_max_elem {

FatehovKMatrixMaxElemMPI::FatehovKMatrixMaxElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FatehovKMatrixMaxElemMPI::ValidationImpl() {
  auto &data = GetInput();
  return (std::get<0>(data) > 0 && std::get<0>(data) <= kMaxRows) &&
         (std::get<1>(data) > 0 && std::get<1>(data) <= kMaxCols) &&
         (std::get<0>(data) * std::get<1>(data) <= kMaxMatrixSize) &&
         (std::get<2>(data).size() <= kMaxMatrixSize &&
          std::get<2>(data).size() == std::get<0>(data) * std::get<1>(data)) &&
         (!std::get<2>(data).empty());
}

bool FatehovKMatrixMaxElemMPI::PreProcessingImpl() {
  return true;
}

bool FatehovKMatrixMaxElemMPI::RunImpl() {
  auto &data = GetInput();
  size_t rows = std::get<0>(data);
  size_t columns = std::get<1>(data);
  std::vector<double> matrix = std::get<2>(data);

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  size_t total_elems = rows * columns;
  size_t elems_per_proc = total_elems / world_size;
  size_t remainder = total_elems % world_size;

  double local_max = -std::numeric_limits<double>::max();
  size_t start = (world_rank * elems_per_proc) + std::min(world_rank, static_cast<int>(remainder));
  size_t end = start + elems_per_proc + (std::cmp_less(world_rank, remainder) ? 1 : 0);

  for (size_t i = start; i < end; i++) {
    local_max = std::max(matrix[i], local_max);
  }

  double global_max = NAN;

  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;

  return true;
}

bool FatehovKMatrixMaxElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_matrix_max_elem
