#include "agafonov_i_sentence_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "agafonov_i_sentence_count/common/include/common.hpp"

namespace agafonov_i_sentence_count {

SentenceCountMPI::SentenceCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SentenceCountMPI::ValidationImpl() {
  return true;
}

bool SentenceCountMPI::PreProcessingImpl() {
  return true;
}

bool SentenceCountMPI::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int total_length = 0;
  if (world_rank == 0) {
    total_length = static_cast<int>(GetInput().length());
  }
  MPI_Bcast(&total_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_length == 0) {
    GetOutput() = 0;
    return true;
  }

  std::string text;
  if (world_rank == 0) {
    text = GetInput();
  } else {
    text.resize(total_length);
  }
  MPI_Bcast(const_cast<char *>(text.data()), total_length, MPI_CHAR, 0, MPI_COMM_WORLD);

  int chunk_size = total_length / world_size;
  int remainder = total_length % world_size;
  int start = (world_rank * chunk_size) + std::min(world_rank, remainder);
  int end = start + chunk_size + (world_rank < remainder ? 1 : 0);

  int local_count = 0;
  bool in_word = false;

  if (start > 0) {
    in_word = (std::isalnum(static_cast<unsigned char>(text[start - 1])) != 0);
  }

  for (int i = start; i < end; ++i) {
    auto c = static_cast<unsigned char>(text[i]);
    if (std::isalnum(c) != 0) {
      in_word = true;
    } else if (in_word && (c == '.' || c == '!' || c == '?')) {
      if (c == '.' && i + 1 < total_length && text[i + 1] == '.') {
        continue;
      }
      local_count++;
      in_word = false;
    }
  }

  int global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (std::isalnum(static_cast<unsigned char>(text[total_length - 1])) != 0) {
    global_count++;
  }

  GetOutput() = global_count;
  return true;
}

bool SentenceCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace agafonov_i_sentence_count
