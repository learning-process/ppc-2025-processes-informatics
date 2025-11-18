#include "luzan_e_matrix_rows_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace luzan_e_matrix_rows_sum {

LuzanEMatrixRowsSumMPI::LuzanEMatrixRowsSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LuzanEMatrixRowsSumMPI::ValidationImpl() {
  int height = std::get<1>(GetInput());
  int width = std::get<2>(GetInput());

  return std::get<0>(GetInput()).size() == static_cast<size_t>(height * width) && height != 0 && width != 0;
}

bool LuzanEMatrixRowsSumMPI::PreProcessingImpl() {
  int height = std::get<1>(GetInput());
  GetOutput().resize(height);
  for (int row = 0; row < height; row++) {
    GetOutput()[row] = 0;
  }
  return true;
}

bool LuzanEMatrixRowsSumMPI::RunImpl() {
  const int height = std::get<1>(GetInput());
  const int width = std::get<2>(GetInput());
  const std::tuple_element_t<0, InType> &mat = std::get<0>(GetInput());
  OutType &sum_vec = GetOutput();
  OutType part_sum_vec = GetOutput();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rest = height % size;
  int rows_per_proc = height / size;
  int begin = rank * rows_per_proc + (rest > rank ? rank : rest);
  int end = begin + rows_per_proc + (rest > rank ? 1 : 0);

  for (int i = begin; i < end; i++) {
    for (int s = 0; s < width; s++) {
      part_sum_vec[i] += mat[width * i + s];
    }
  }

  MPI_Reduce(part_sum_vec.data(), sum_vec.data(), height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(sum_vec.data(), height, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool LuzanEMatrixRowsSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_rows_sum
