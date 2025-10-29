#include "akimov_i_words_string_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>
#include <cstring>

#include "akimov_i_words_string_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace akimov_i_words_string_count {

AkimovIWordsStringCountMPI::AkimovIWordsStringCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkimovIWordsStringCountMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return (!GetInput().empty()) && (GetOutput() == 0);
  }
  return true;
}

bool AkimovIWordsStringCountMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  local_space_count_ = 0;
  global_space_count_ = 0;
  word_count_ = 0;
  local_buffer_.clear();
  input_buffer_.clear();

  std::size_t total = 0;
  if (rank == 0) {
    input_buffer_ = GetInput();
    total = input_buffer_.size();
  }

  int base = 0;
  int remainder = 0;
  if (rank == 0) {
    base = static_cast<int>(total / static_cast<std::size_t>(size));
    remainder = static_cast<int>(total % static_cast<std::size_t>(size));
  }
  MPI_Bcast(&base, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&remainder, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int my_count = base + ((rank < remainder) ? 1 : 0);
  local_buffer_.resize(my_count);

  if (rank == 0) {
    std::size_t offset = 0;
    for (int proc = 0; proc < size; ++proc) {
      int count = base + ((proc < remainder) ? 1 : 0);
      if (count == 0) {

      } else if (proc == 0) {
        std::memcpy(local_buffer_.data(), input_buffer_.data() + offset, static_cast<std::size_t>(count));
      } else {
        MPI_Send(input_buffer_.data() + offset, count, MPI_CHAR, proc, 0, MPI_COMM_WORLD);
      }
      offset += static_cast<std::size_t>(count);
    }
  } else {
    if (my_count > 0) {
      MPI_Recv(local_buffer_.data(), my_count, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {

    }
  }

  return true;
}

bool AkimovIWordsStringCountMPI::RunImpl() {
  local_space_count_ = 0;
  for (char c : local_buffer_) {
    if (c == ' ') {
      ++local_space_count_;
    }
  }

  MPI_Reduce(&local_space_count_, &global_space_count_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (GetInput().empty()) {
      word_count_ = 0;
    } else {
      const InType &buf = GetInput();
      bool in_word = false;
      word_count_ = 0;
      for (char c : buf) {
        if (c != ' ' && !in_word) {
          in_word = true;
          ++word_count_;
        } else if (c == ' ' && in_word) {
          in_word = false;
        }
      }
    }
  }

  return true;
}

bool AkimovIWordsStringCountMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput() = word_count_;
  }
  return true;
}

}  // namespace akimov_i_words_string_count
