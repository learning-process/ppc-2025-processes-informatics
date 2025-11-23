#include "lukin_i_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "lukin_i_elem_vec_sum/common/include/common.hpp"

namespace lukin_i_elem_vec_sum {

LukinIElemVecSumMPI::LukinIElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LukinIElemVecSumMPI::ValidationImpl() {
  return (static_cast<int>(GetInput().size()) != 0);
}

bool LukinIElemVecSumMPI::PreProcessingImpl() {
  vec_size_ = static_cast<int>(GetInput().size());
  return true;
}

bool LukinIElemVecSumMPI::RunImpl() {
  int proc_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (proc_count > vec_size_) {
    if (rank == 0) {
      GetOutput() = std::accumulate(GetInput().begin(), GetInput().end(), 0LL);
    } else {
      GetOutput() = 0;
    }
    OutType output = GetOutput();
    MPI_Bcast(&output, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    GetOutput() = output;
    return true;
  }

  const int part = vec_size_ / proc_count;
  const int reminder = vec_size_ % proc_count;

  int start = (part * rank) + ((rank < reminder) ? rank : reminder);
  int end = start + part + ((rank < reminder) ? 1 : 0);

  OutType local_sum = std::accumulate(GetInput().begin() + start, GetInput().begin() + end, 0LL);

  OutType global_sum = 0LL;

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;

  return true;
}

bool LukinIElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_elem_vec_sum
