#include "fatehov_k_matrix_max_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "fatehov_k_matrix_max_elem/common/include/common.hpp"
#include "util/include/util.hpp"

namespace fatehov_k_matrix_max_elem {

FatehovKMatrixMaxElemMPI::FatehovKMatrixMaxElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FatehovKMatrixMaxElemMPI::ValidationImpl() {
  auto &data = GetInput();
  return (std::get<0>(data) > 0 && std::get<0>(data) <= MAX_ROWS) &&
         (std::get<1>(data) > 0 && std::get<1>(data) <= MAX_COLS) &&
         (std::get<0>(data) * std::get<1>(data) <= MAX_MATRIX_SIZE) &&
         (std::get<2>(data).size() <= MAX_MATRIX_SIZE &&
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

  int world_rank;
  int world_size;
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
      elem_counts[i] = elem_per_process;
      if (i < (int)remainder_elems) {
        elem_counts[i]++;
      }
      elem_displacements[i] = current_displacement;
      current_displacement += elem_counts[i];
    }
  }

  int my_element_count;
  MPI_Scatter(elem_counts.data(), 1, MPI_INT, &my_element_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> local_elements(my_element_count);
  MPI_Scatterv(matrix.data(), elem_counts.data(), elem_displacements.data(), MPI_DOUBLE, local_elements.data(),
               my_element_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_max = -std::numeric_limits<double>::max();
  for (size_t i = 0; i < local_elements.size(); i++) {
    if (local_elements[i] > local_max) {
      local_max = local_elements[i];
    }
  }

  double global_max;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool FatehovKMatrixMaxElemMPI::PostProcessingImpl() {
  return true;
}

}  // namespace fatehov_k_matrix_max_elem
