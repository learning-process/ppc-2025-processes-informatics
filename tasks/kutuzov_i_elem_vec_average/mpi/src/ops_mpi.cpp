#include "kutuzov_i_elem_vec_average/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
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
  return GetInput().size() > 0;
}

bool KutuzovIElemVecAverageMPI::PreProcessingImpl() {
  return true;
}

bool KutuzovIElemVecAverageMPI::RunImpl() {

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double global_sum = 0.0;
  
  size_t num_threads = ppc::util::GetNumThreads();
  size_t batch_size = GetInput().size() / num_threads;

  std::vector<double> recv_buffer(batch_size);

  MPI_Scatter(GetInput().data(), GetInput().size(), MPI_DOUBLE, recv_buffer.data(), batch_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double sum = 0.0;
  for (size_t i = 0; i < batch_size; i++)
    sum += recv_buffer[i];

  MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
    GetOutput() = global_sum / GetInput().size();

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KutuzovIElemVecAverageMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_elem_vec_average
