#include "kutuzov_i_elem_vec_average/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
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

  int total_elements_num = 0;
  if (rank == 0) {
    total_elements_num = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_elements_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int batch_size = 0;
  batch_size = total_elements_num / num_processes;

  if (batch_size > 0) {
    double* local_buffer = new double[batch_size];
    MPI_Scatter(GetInput().data(), batch_size, MPI_DOUBLE, local_buffer, batch_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double sum = 0.0;
    for (int i = 0; i < batch_size; i++) {
      sum += input[rank * batch_size + i];
    }

    MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    delete[] local_buffer;
  }

  if (rank == 0) {
    if (num_processes * batch_size < total_elements_num) {
      for (int i = num_processes * batch_size; i < total_elements_num; i++) {
        global_sum += input[i];
      }
    }
    result = global_sum / static_cast<double>(total_elements_num);
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = result;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KutuzovIElemVecAverageMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_elem_vec_average
