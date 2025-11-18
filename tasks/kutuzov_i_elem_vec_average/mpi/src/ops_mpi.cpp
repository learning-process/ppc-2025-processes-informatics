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

  double result = 0.0;
  double global_sum = 0.0;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_processes = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  int batch_size = static_cast<int>(input.size()) / num_processes;
  MPI_Bcast(&batch_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (batch_size > 0) {
    double *recv_buffer = new double[batch_size];
    MPI_Scatter(input.data(), batch_size, MPI_DOUBLE, recv_buffer, batch_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double sum = 0.0;
    for (int i = 0; i < batch_size; i++) {
      sum += recv_buffer[i];
    }

    MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    delete[] recv_buffer;
  }

  if (rank == 0) {
    for (int i = num_processes * batch_size; i < static_cast<int>(input.size()); i++) {
      global_sum += input[i];
    }

    result = global_sum / static_cast<double>(input.size());
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
