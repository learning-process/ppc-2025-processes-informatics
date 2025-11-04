#include "baldin_a_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cctype>
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
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::string &input = GetInput();

  if (input.size() == 0) {
    GetOutput() = 0;
    return true;
  }

  auto is_word_char = [](char c) -> bool {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
  };

  if (world_size > static_cast<int>(input.size())) {
    if (rank == 0) {
      size_t count = 0;
      bool in_word = false;
      for (char c : input) {
        if (is_word_char(c)) {
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
    MPI_Bcast(static_cast<void *>(&GetOutput()), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    return true;
  }

  int total_len = static_cast<int>(input.size());
  int rem = total_len % world_size;
  if (rem != 0) {
    input.append(world_size - rem, ' ');
  }
  int part = input.size() / world_size;

  std::vector<char> local_buf(part);
  MPI_Scatter(input.data(), part, MPI_CHAR, local_buf.data(), part, MPI_CHAR, 0, MPI_COMM_WORLD);

  size_t local_cnt = 0;
  bool in_word = false;
  for (int i = 0; i < part; i++) {
    if (is_word_char(local_buf[i])) {
      if (!in_word) {
        in_word = true;
        local_cnt++;
      }
    } else {
      in_word = false;
    }
  }

  if (rank > 0) {
    char first_char = local_buf[0];
    MPI_Send(&first_char, 1, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD);
  }

  if (rank < world_size - 1) {
    char next_first_char;
    MPI_Recv(&next_first_char, 1, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (is_word_char(local_buf.back()) && is_word_char(next_first_char)) {
      local_cnt--;
    }
  }

  size_t global_cnt = 0;
  MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    GetOutput() = global_cnt;
  }

  MPI_Bcast(static_cast<void *>(&GetOutput()), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  return true;
}

bool BaldinAWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_word_count
