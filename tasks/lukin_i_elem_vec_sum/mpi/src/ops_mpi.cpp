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
  auto &input = GetInput();
  int vec_size = static_cast<int>(input.size());

  int proc_count, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int part = vec_size / proc_count;
  const int reminder = vec_size % proc_count;

  if (proc_count > vec_size) {
    int global_sum = 0;
    if (rank == 0) {
      global_sum = std::accumulate(input.begin(), input.end(), 0);
    }
    MPI_Bcast(&global_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = global_sum;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int local_size = (rank != proc_count - 1) ? part : part + reminder;
  std::vector<int> local_vector(local_size);

  if (rank == 0) {
    for (int i = 0; i < local_size; i++) {
      local_vector[i] = input[i];
    }
    for (int i = 1; i < proc_count; i++) {
      int send_size = (i != proc_count - 1) ? part : part + reminder;
      MPI_Send(input.data() + i * part, send_size, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_vector.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int local_sum = 0;
  if (local_size != 0) {
    local_sum = std::accumulate(local_vector.begin(), local_vector.end(), 0);
  }

  int global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = global_sum;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LukinIElemVecSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_elem_vec_sum
