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
  size_t elems = rows * columns;
  size_t elem_per_process = elems / world_size;
  size_t remainder_elems = elems % world_size;

  std::vector<int> elem_counts(world_size);
  std::vector<int> elem_displacements(world_size);

  if (world_rank == 0) {
    int current_displacement = 0;
    for (int i = 0; i < world_size; i++) {
      elem_counts[i] = static_cast<int>(elem_per_process);
      if (std::cmp_less(i, remainder_elems)) {
        elem_counts[i]++;
      }
      elem_displacements[i] = current_displacement;
      current_displacement += elem_counts[i];
    }
  }

  int my_element_count = 0;
  MPI_Scatter(elem_counts.data(), 1, MPI_INT, &my_element_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> local_elements(my_element_count);
  MPI_Scatterv(matrix.data(), elem_counts.data(), elem_displacements.data(), MPI_DOUBLE, local_elements.data(),
               my_element_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_max = -std::numeric_limits<double>::max();
  for (double element : local_elements) {
    local_max = std::max(element, local_max);
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
