#include "../include/char_freq_mpi.hpp"

#include <mpi.h>

#include "../../common/include/common.hpp"

namespace shekhirev_v_char_freq_mpi {

CharFreqMPI::CharFreqMPI(const shekhirev_v_char_freq_seq::InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool CharFreqMPI::ValidationImpl() {
  return true;
}

bool CharFreqMPI::PreProcessingImpl() {
  int process_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  int str_size = 0;
  if (process_rank == 0) {
    str_size = static_cast<int>(GetInput().str.size());
  }

  MPI_Bcast(&str_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (process_rank != 0) {
    GetInput().str.resize(str_size);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  MPI_Bcast(const_cast<char *>(GetInput().str.data()), str_size, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&GetInput().target, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  return true;
}

bool CharFreqMPI::RunImpl() {
  int process_rank = 0;
  int process_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  const auto &str = GetInput().str;
  char target = GetInput().target;
  int n = static_cast<int>(str.size());

  const int base_n = n / process_count;
  const int remainder = n % process_count;

  int start_index = 0;
  if (process_rank < remainder) {
    start_index = process_rank * (base_n + 1);
  } else {
    start_index = (remainder * (base_n + 1)) + ((process_rank - remainder) * base_n);
  }

  int local_n = base_n + (process_rank < remainder ? 1 : 0);
  int end_index = start_index + local_n;

  int local_count = 0;
  if (local_n > 0 && start_index < n) {
    for (int i = start_index; i < end_index; ++i) {
      if (str[i] == target) {
        local_count++;
      }
    }
  }

  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (process_rank == 0) {
    GetOutput() = global_count;
  }

  return true;
}

bool CharFreqMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shekhirev_v_char_freq_mpi
