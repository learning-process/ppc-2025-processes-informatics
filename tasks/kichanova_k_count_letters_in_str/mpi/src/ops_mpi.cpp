#include "kichanova_k_count_letters_in_str/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cctype>
#include <string>

#include "kichanova_k_count_letters_in_str/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_count_letters_in_str {

KichanovaKCountLettersInStrMPI::KichanovaKCountLettersInStrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KichanovaKCountLettersInStrMPI::ValidationImpl() {
  return !GetInput().empty() && (GetOutput() == 0);
}

bool KichanovaKCountLettersInStrMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KichanovaKCountLettersInStrMPI::RunImpl() {
  auto input_str = GetInput();
  if (input_str.empty()) {
    return false;
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_length = input_str.length();
  int chunk_size = total_length / size;
  int remainder = total_length % size;

  int start_index = rank * chunk_size + std::min(rank, remainder);
  int end_index = start_index + chunk_size + (rank < remainder ? 1 : 0);

  int local_count = 0;
  for (int i = start_index; i < end_index && i < total_length; i++) {
    if (std::isalpha(static_cast<unsigned char>(input_str[i]))) {
      local_count++;
    }
  }

  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_count;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    return GetOutput() > 0;
  }
  return true;

}

bool KichanovaKCountLettersInStrMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace kichanova_k_count_letters_in_str
