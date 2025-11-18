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
  size_t height = std::get<1>(GetInput());
  size_t width = std::get<2>(GetInput());

  return std::get<0>(GetInput()).size() == (height * width) && height != 0 && width != 0;
}

bool LuzanEMatrixRowsSumMPI::PreProcessingImpl() {
  size_t height = std::get<1>(GetInput());
  GetOutput().resize(height);
  for (size_t row = 0; row < height; row++) {
    GetOutput()[row] = 0;
  }
  return true;
}

bool LuzanEMatrixRowsSumMPI::RunImpl() {
  const size_t height = std::get<1>(GetInput());
  const size_t width = std::get<2>(GetInput());
  const std::tuple_element_t<0, InType> &mat = std::get<0>(GetInput());
  OutType &sum_vec = GetOutput();
  OutType part_sum_vec = GetOutput();

  int rank_, size_;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  size_t rank = static_cast<size_t>(rank_);
  size_t size = static_cast<size_t>(size_);
  size_t rest = height % size;
  size_t rows_per_proc = height / size;
  size_t begin = rank * rows_per_proc + (rest > rank ? rank : rest);
  size_t end = begin + rows_per_proc + (rest > rank ? 1 : 0);

  for (size_t i = begin; i < end; i++) {
    for (size_t s = 0; s < width; s++) {
      part_sum_vec[i] += mat[width * i + s];
    }
  }

  MPI_Reduce(part_sum_vec.data(), sum_vec.data(), height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(sum_vec.data(), height, MPI_INT, 0, MPI_COMM_WORLD);
  // MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LuzanEMatrixRowsSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_rows_sum
