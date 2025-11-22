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
  }
  else {
    GetInput() = in_reduced;
  }
}

bool LuzanEMatrixRowsSumMPI::ValidationImpl() {
  size_t height = std::get<1>(GetInput());
  size_t width = std::get<2>(GetInput());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t mat_size = (rank == 0 ? height * width : 0);
  return std::get<0>(GetInput()).size() == mat_size && height != 0 && width != 0;
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
  std::tuple_element_t<0, InType> mat;
  
	int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    mat = std::get<0>(GetInput());
  }

  int rest = height % size;
  std::vector<int> shift(size);
  std::vector<int> per_proc(size, height / size);

  shift[0] = 0;
  if (rest != 0) {
    per_proc[0]++;
    rest--;
  }
  for (int i = 1; i < size; i++) {
    if (rest) {
      per_proc[i]++;
      rest--;
    }
    shift[i] = per_proc[i - 1] + shift[i - 1];
  }

  std::vector<int> recv(per_proc[rank] * width);
 
  for (int i = 0; i < size; i++) {
    per_proc[i] *= width;
    shift[i] *= width;
  }
  
  MPI_Scatterv(mat.data(), per_proc.data(), shift.data(), MPI_INT, recv.data(), per_proc[rank], MPI_INT, 0, MPI_COMM_WORLD);
  mat.clear();
  std::vector<int> sum(height);
  int abs_shift = static_cast<int>(shift[rank] / width);
  int rows_to_calc = static_cast<int>(per_proc[rank] / width);
  for (int row = 0; row < rows_to_calc; row++)
    for (size_t col = 0; col < width; col++)
      sum[row + abs_shift] += recv[row * width + col];
 
  std::vector<int> fin_sum(height);
	MPI_Reduce(sum.data(), fin_sum.data(), sum.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(fin_sum.data(), static_cast<int>(height), MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = fin_sum;
  return true;
}

bool LuzanEMatrixRowsSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luzan_e_matrix_rows_sum
