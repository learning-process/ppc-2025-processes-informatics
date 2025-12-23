#include "agafonov_i_sentence_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <vector>

namespace agafonov_i_sentence_count {

SentenceCountMPI::SentenceCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SentenceCountMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool SentenceCountMPI::PreProcessingImpl() {
  return GetOutput() >= 0;
}

bool SentenceCountMPI::RunImpl() {
  const std::string &text = GetInput();

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int total_length = static_cast<int>(text.length());
  int chunk_size = total_length / world_size;
  int remainder = total_length % world_size;

  int start = world_rank * chunk_size + std::min(world_rank, remainder);
  int end = start + chunk_size + (world_rank < remainder ? 1 : 0);

  if (start >= total_length) {
    int local_count = 0;
    int global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = global_count;
    return true;
  }

  int local_count = 0;
  bool local_in_sentence = false;

  if (world_rank > 0 && start > 0) {
    char prev_char = text[start - 1];
    local_in_sentence = std::isalnum(static_cast<unsigned char>(prev_char));
  }

  for (int i = start; i < end && i < total_length; ++i) {
    char c = text[i];

    if (std::isalnum(static_cast<unsigned char>(c))) {
      local_in_sentence = true;
    } else if (c == '.' && local_in_sentence) {
      if (i + 1 < total_length && text[i + 1] == '.') {
        continue;
      }
      local_count++;
      local_in_sentence = false;
    } else if ((c == '!' || c == '?') && local_in_sentence) {
      local_count++;
      local_in_sentence = false;
    }
  }

  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    for (int i = 1; i < world_size; ++i) {
      int boundary_idx = (i * chunk_size + std::min(i, remainder)) - 1;

      if (boundary_idx >= 0 && boundary_idx < total_length - 1) {
        char left_char = text[boundary_idx];
        char right_char = text[boundary_idx + 1];

        if (std::isalnum(static_cast<unsigned char>(left_char)) &&
            (right_char == '.' || right_char == '!' || right_char == '?')) {
          if (!(right_char == '.' && boundary_idx + 2 < total_length && text[boundary_idx + 2] == '.')) {
            global_count--;
          }
        }
      }
    }

    if (total_length > 0) {
      char last_char = text[total_length - 1];
      if (std::isalnum(static_cast<unsigned char>(last_char))) {
        global_count++;
      }
    }
  }

  MPI_Bcast(&global_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_count;

  return true;
}

bool SentenceCountMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace agafonov_i_sentence_count
