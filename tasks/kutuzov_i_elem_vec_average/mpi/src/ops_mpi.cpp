#include "kutuzov_i_elem_vec_average/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "kutuzov_i_elem_vec_average/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kutuzov_i_elem_vec_average {

KutuzovIElemVecAverageMPI::KutuzovIElemVecAverageMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool KutuzovIElemVecAverageMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool KutuzovIElemVecAverageMPI::PreProcessingImpl() {
  return true;
}

bool KutuzovIElemVecAverageMPI::RunImpl() {
  const auto &input = GetInput();
  GetOutput() = 0.0;

  double result = 0.0;
  double global_sum = 0.0;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  int batch_size = 0;
  batch_size = static_cast<int>(input.size()) / num_processes;

  if (batch_size > 0) {
    double sum = 0.0;
    for (int i = 0; i < batch_size; i++) {
      sum += input[rank * batch_size + i];
    }
    std::cout << input.size() << " : " << rank << " " << sum << std::endl;
    MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    if (num_processes * batch_size < static_cast<int>(input.size())) {
      for (int i = num_processes * batch_size; i < static_cast<int>(input.size()); i++) {
        global_sum += input[i];
      }
    }
    std::cout << input.size() << " : " << "Global Sum: " << global_sum << std::endl;
    result = global_sum / static_cast<double>(input.size());
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = result;

  std::cout << input.size() << " : " << "Answer: " << rank << " " << GetOutput() << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KutuzovIElemVecAverageMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_elem_vec_average
