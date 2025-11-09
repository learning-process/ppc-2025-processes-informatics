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
  return true;
}

bool LukinIElemVecSumMPI::PreProcessingImpl() {
  vec_size = static_cast<int>(GetInput().size());
  return true;
}

bool LukinIElemVecSumMPI::RunImpl() {
  if (vec_size == 0) {
    GetOutput() = 0;
    return true;
  }

  int proc_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (proc_count > vec_size) {
    if (rank == 0) {
      GetOutput() = std::accumulate(GetInput().begin(), GetInput().end(), 0);
    } else {
      GetOutput() = 0;
    }
    MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  std::vector<int> sendcounts(proc_count, 0);
  std::vector<int> offsets(proc_count, 0);

  int local_size = 0;

  const int part = vec_size / proc_count;
  const int reminder = vec_size % proc_count;

  int offset = 0;
  for (int i = 0; i < proc_count; i++) {
    sendcounts[i] = part + (i < reminder ? 1 : 0);
    offsets[i] = offset;
    offset += sendcounts[i];
  }

  local_size = sendcounts[rank];
  std::vector<int> local_vec(local_size);

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), offsets.data(), MPI_INT, local_vec.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  OutType local_sum = std::accumulate(local_vec.begin(), local_vec.end(), 0);

  OutType global_sum = 0;

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;

  return true;
}

bool LukinIElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_elem_vec_sum
