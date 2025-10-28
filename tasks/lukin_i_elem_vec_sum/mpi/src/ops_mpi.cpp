#include "lukin_i_elem_vec_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "lukin_i_elem_vec_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace lukin_i_elem_vec_sum {

LukinIElemVecSumMPI::LukinIElemVecSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LukinIElemVecSumMPI::ValidationImpl() {
  const auto &vec = GetInput();
  return !vec.empty();
}

bool LukinIElemVecSumMPI::PreProcessingImpl() {
  return true;
}

bool LukinIElemVecSumMPI::RunImpl() {
  const int vec_size = GetInput().size();
  if (vec_size == 0) {
    GetOutput() = 0;
    return true;
  }

  int proc_count, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (proc_count == 1) {
    GetOutput() = std::accumulate(GetInput().begin(), GetInput().end(), 0);
    return true;
  }

  const int part = vec_size / proc_count;
  const int reminder = vec_size % proc_count;

  std::vector<int> sendcounts(proc_count, 0);
  std::vector<int> offsets(proc_count, 0);

  for (int i = 0; i < proc_count; i++) {
    sendcounts[i] = part + (i < reminder ? 1 : 0);
    offsets[i] = (i == 0) ? 0 : sendcounts[i - 1] + offsets[i - 1];
  }

  std::vector<int> local_vec(sendcounts[rank], 0);
  MPI_Scatterv(GetInput().data(), sendcounts.data(), offsets.data(), MPI_INT, local_vec.data(), sendcounts[rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  int local_sum = std::accumulate(local_vec.begin(), local_vec.end(), 0);
  int global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;
  return true;
}

bool LukinIElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_elem_vec_sum
