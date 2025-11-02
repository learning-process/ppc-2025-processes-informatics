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
  int rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::string &input = GetInput();

  if (input.size() == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    MPI_Bcast(static_cast<void *>(&GetOutput()), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    return true;
  }

  if (world_size > static_cast<int>(input.size())) {
    if (rank == 0) {
      size_t count = 0;
      bool in_word = false;
      for (char c : input) {
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
    MPI_Bcast(static_cast<void *>(&GetOutput()), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    return true;
  }

  int total_len = static_cast<int>(input.size());
  int rem = total_len % world_size;
  if (rem != 0) {
    input.append(world_size - rem + 1, ' ');
  }
  int part = input.size() / world_size;

  std::cout << "+" << '\n';
  std::vector<int> send_counts(world_size), displs(world_size);

  if (rank == 0) {
    int offset = 0;
    for (int i = 0; i < world_size; i++) {
      send_counts[i] = part + 1;
      displs[i] = offset;
      offset += part;
    }
  }
  std::cout << "++" << '\n';
  // if (rank == 0) {
  //   std::cout << total_len << '\n';
  //   for (int i = 0; i < world_size; i++) {
  //     std::cout << send_counts[i] << ' ';
  //   }
  //   std::cout << '\n';
  // }

  std::vector<char> local_buf(part + 1);

  MPI_Scatterv(rank == 0 ? input.data() : nullptr, rank == 0 ? send_counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_CHAR, local_buf.data(), part + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  std::cout << "+++" << '\n';
  size_t local_cnt = 0;
  bool in_word = false;
  for (int i = 0; i < part; i++) {
    if (std::isalnum(local_buf[i]) || local_buf[i] == '-' || local_buf[i] == '_') {
      if (!in_word) {
        in_word = true;
        local_cnt++;
      }
    } else {
      in_word = false;
    }
  }

  if (in_word && (std::isalnum(local_buf[part]) || local_buf[part] == '-' || local_buf[part] == '_')) {
    local_cnt--;
  }
  std::cout << "++++" << '\n';
  // std::cout << "RANK: " << rank << ' ' << local_cnt << '\n';
  // for (int i = 0; i < part; i++) {
  //   std::cout << local_buf[i];
  // }
  // std::cout << '\n';

  size_t global_cnt = 0;
  MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  std::cout << "+++++" << '\n';
  if (rank == 0) {
    GetOutput() = global_cnt;
    // std::cout << "ASWERRRRR: " << global_cnt << '\n';
  }

  MPI_Bcast(static_cast<void *>(&GetOutput()), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  std::cout << "++++++" << '\n';
  return true;
}

bool BaldinAWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_word_count
