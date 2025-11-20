#include "../include/char_freq_mpi.hpp"

#include <mpi.h>

#include <algorithm>

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

  if (n == 0) {
    if (process_rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  const int delta = n / process_count;
  const int remainder = n % process_count;

  int start_index = 0;
  int end_index = 0;

  if (process_rank < remainder) {
    start_index = process_rank * (delta + 1);
    end_index = start_index + delta + 1;
  } else {
    start_index = (process_rank * delta) + remainder;
    end_index = start_index + delta;
  }

  int local_count = 0;
  if (start_index < end_index) {
    local_count = static_cast<int>(std::count(str.begin() + start_index, str.begin() + end_index, target));
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
