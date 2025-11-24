#include "volkov_a_count_word_line/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace volkov_a_count_word_line {

VolkovACountWordLineMPI::VolkovACountWordLineMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VolkovACountWordLineMPI::ValidationImpl() {
  return true;
}

bool VolkovACountWordLineMPI::PreProcessingImpl() {
  return true;
}

bool VolkovACountWordLineMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto is_token_char = [](char c) -> bool {
    const bool is_alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    const bool is_digit = (c >= '0' && c <= '9');
    const bool is_special = (c == '-' || c == '_');
    return is_alpha || is_digit || is_special;
  };

  int total_len = 0;
  if (rank == 0) {
    total_len = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_len == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  int base_chunk = total_len / world_size;
  int remainder = total_len % world_size;

  int my_count = base_chunk + (rank < remainder ? 1 : 0);

  std::vector<int> send_counts;
  std::vector<int> displs;

  if (rank == 0) {
    send_counts.resize(world_size);
    displs.resize(world_size);
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
      displs[i] = offset;
      offset += send_counts[i];
    }
  }

  std::vector<char> local_data(my_count);
  const char *send_buf = (rank == 0) ? GetInput().data() : nullptr;

  MPI_Scatterv(send_buf, (rank == 0 ? send_counts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_CHAR,
               local_data.data(), my_count, MPI_CHAR, 0, MPI_COMM_WORLD);

  char prev_char = ' ';
  char my_last = (my_count > 0) ? local_data.back() : ' ';

  int left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  int right_neighbor = (rank == world_size - 1) ? MPI_PROC_NULL : rank + 1;

  MPI_Sendrecv(&my_last, 1, MPI_CHAR, right_neighbor, 0, &prev_char, 1, MPI_CHAR, left_neighbor, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  int local_words = 0;
  int idx = 0;

  if (my_count > 0 && is_token_char(local_data[0])) {
    if (is_token_char(prev_char)) {
      while (idx < my_count && is_token_char(local_data[idx])) {
        idx++;
      }
    }
  }

  while (idx < my_count) {
    while (idx < my_count && !is_token_char(local_data[idx])) {
      idx++;
    }

    if (idx < my_count) {
      local_words++;
      while (idx < my_count && is_token_char(local_data[idx])) {
        idx++;
      }
    }
  }

  int total_words = 0;
  MPI_Reduce(&local_words, &total_words, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = total_words;
  }

  return true;
}

bool VolkovACountWordLineMPI::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_count_word_line