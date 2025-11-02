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

  std::string input_local = GetInput();

  if (rank == 0 && input_local.empty()) {
    GetOutput() = 0;
  }
  int is_empty = (rank == 0 && input_local.empty()) ? 1 : 0;
  MPI_Bcast(&is_empty, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (is_empty) {
    return true;
  }

  int total_len = 0;
  if (rank == 0) {
    total_len = static_cast<int>(input_local.size());
  }

  MPI_Bcast(&total_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int padded_len = total_len;
  if (rank == 0) {
    int rem = total_len % world_size;
    int pad = (rem == 0) ? 0 : (world_size - rem);
    input_local.append(pad + 1, ' ');
    padded_len = static_cast<int>(input_local.size());
  }

  MPI_Bcast(&padded_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int part = padded_len / world_size;
  int recv_count = (rank == world_size - 1) ? (padded_len - part * (world_size - 1)) : (part + 1);

  std::vector<int> send_counts;
  std::vector<int> displs;
  if (rank == 0) {
    displs.assign(world_size, 0);
    send_counts.assign(world_size, 0);
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      displs[i] = offset;
      offset += part;
    }
    for (int i = 0; i < world_size; ++i) {
      if (i < world_size - 1) {
        send_counts[i] = part + 1;
      } else {
        send_counts[i] = padded_len - displs[i];
      }
    }
  }

  std::vector<char> local_buf(recv_count);

  MPI_Scatterv(rank == 0 ? input_local.data() : nullptr, rank == 0 ? send_counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_CHAR, local_buf.data(), recv_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  auto is_word_char = [](char c) -> bool {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
  };

  long long local_cnt = 0;
  bool in_word = false;

  int main_len = (rank == world_size - 1) ? (recv_count - 1) : part;
  if (main_len < 0) {
    main_len = 0;
  }
  for (int i = 0; i < main_len; ++i) {
    if (is_word_char(local_buf[i])) {
      if (!in_word) {
        in_word = true;
        ++local_cnt;
      }
    } else {
      in_word = false;
    }
  }
  if (rank < world_size - 1 && main_len > 0) {
    if (is_word_char(local_buf[main_len - 1]) && is_word_char(local_buf[main_len])) {
      --local_cnt;
    }
  }

  long long global_cnt = 0;
  MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = static_cast<long long>(global_cnt);
  }

  long long broadcast_out = (rank == 0) ? global_cnt : 0;
  MPI_Bcast(&broadcast_out, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    GetOutput() = broadcast_out;
  }

  return true;
}

bool BaldinAWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_word_count
