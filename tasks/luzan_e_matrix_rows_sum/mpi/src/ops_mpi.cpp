#include "luzan_e_matrix_rows_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "luzan_e_matrix_rows_sum/common/include/common.hpp"

namespace luzan_e_matrix_rows_sum {

LuzanEMatrixRowsSumMPI::LuzanEMatrixRowsSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = {};

  // saving matrix only if it's rank=0
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> tmp(0);
  InType in_reduced = std::make_tuple(tmp, std::get<1>(in), std::get<2>(in));
  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = in_reduced;
  }
}

bool LuzanEMatrixRowsSumMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }

  int height = std::get<1>(GetInput());
  int width = std::get<2>(GetInput());
  return static_cast<int>(std::get<0>(GetInput()).size()) == (height * width) && height != 0 && width != 0;
}

bool LuzanEMatrixRowsSumMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }

  int height = std::get<1>(GetInput());
  GetOutput().resize(height);
  for (int row = 0; row < height; row++) {
    GetOutput()[row] = 0;
  }
  return true;
}

bool LuzanEMatrixRowsSumMPI::RunImpl() {
  int height = 0;
  int width = 0;
  std::tuple_element_t<0, InType> mat;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> dim(2, 0);
  if (rank == 0) {
    mat = std::get<0>(GetInput());
    height = std::get<1>(GetInput());
    width = std::get<2>(GetInput());
    dim[0] = height;
    dim[1] = width;
  }
  MPI_Bcast(dim.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
  height = dim[0];
  width = dim[1];

  int rest = height % size;
  std::vector<int> shift(size);
  std::vector<int> per_proc(size, height / size);

  shift[0] = 0;
  if (rest != 0) {
    per_proc[0]++;
    rest--;
  }
  for (int i = 1; i < size; i++) {
    if (rest != 0) {
      per_proc[i]++;
      rest--;
    }
    shift[i] = per_proc[i - 1] + shift[i - 1];
  }

  std::vector<int> recv(static_cast<size_t>(per_proc[rank] * width));

  for (int i = 0; i < size; i++) {
    per_proc[i] *= width;
    shift[i] *= width;
  }

  MPI_Scatterv(mat.data(), per_proc.data(), shift.data(), MPI_INT, recv.data(), per_proc[rank], MPI_INT, 0,
               MPI_COMM_WORLD);
  mat.clear();
  std::vector<int> sum(height);
  int abs_shift = static_cast<int>(shift[rank] / width);
  int rows_to_calc = static_cast<int>(per_proc[rank] / width);
  for (int row = 0; row < rows_to_calc; row++) {
    for (int col = 0; col < width; col++) {
      sum[row + abs_shift] += recv[(row * width) + col];
    }
  }

  std::vector<int> fin_sum(height);
  MPI_Reduce(sum.data(), fin_sum.data(), static_cast<int>(sum.size()), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(fin_sum.data(), height, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = fin_sum;
  return true;
}

bool LuzanEMatrixRowsSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_rows_sum
