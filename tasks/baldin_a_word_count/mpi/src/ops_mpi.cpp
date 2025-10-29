#include "baldin_a_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "baldin_a_word_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_word_count {

BaldinAWordCountMPI::BaldinAWordCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BaldinAWordCountMPI::ValidationImpl() {
  return true;
}

bool BaldinAWordCountMPI::PreProcessingImpl() {
  return true;
}

bool BaldinAWordCountMPI::RunImpl() {
  int rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  //написать auto input = GetInput();

  if (GetInput().size() == 0) {
    if (rank == 0) GetOutput() = 0;
    return true;
  }

  if (world_size > static_cast<int>(GetInput().size())) {
    if (rank == 0) {
      int count = 0;
      bool in_word = false;
      for (char c : GetInput()) {
        if (std::isalnum(c) || c == '-' || c == '_') {
            if (!in_word) {
                in_word = true;
                count++;
            }
        } else {
            in_word = false;
        }
      }
      GetOutput() = count;
    }
    MPI_Bcast(static_cast<void*>(&GetOutput()), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }
  
  std::vector<int> send_counts(world_size), displs(world_size);
  int part = (GetInput().size() + world_size - 1) / world_size;

  if (rank == 0) {
    for (int i = 0; i < world_size; i++) {
      displs[i] = i * part;
      send_counts[i] = (i == world_size - 1) ? (GetInput().size() - displs[i]) : part + 1;
    }
  }

  std::vector<char> local_buf(part + 1);
  MPI_Scatterv(GetInput().data(), send_counts.data(), displs.data(), MPI_CHAR, local_buf.data(), part + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_cnt = 0;
  bool in_word = false;
  for (int i = 0; i < part; i++) {
      if (isalnum(local_buf[i]) || local_buf[i] == '-' || local_buf[i] == '_') {
          if (!in_word) {
              in_word = true;
              local_cnt++;
          }
      } else {
          in_word = false;
      }
  }
  if ((isalnum(local_buf[part - 1]) || local_buf[part - 1] == '-' || local_buf[part - 1] == '_') && (isalnum(local_buf[part]) || local_buf[part] == '-' || local_buf[part] == '_')) {
      local_cnt--;
  }

  int global_cnt = 0;
  MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_cnt;
  }

  MPI_Bcast(static_cast<void*>(&GetOutput()), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
  
}

bool BaldinAWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_word_count
