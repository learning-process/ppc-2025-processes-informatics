#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {

SizovDStringMismatchCountMPI::SizovDStringMismatchCountMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0;
}

bool SizovDStringMismatchCountMPI::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);
  return !a.empty() && a.size() == b.size();
}

bool SizovDStringMismatchCountMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  str_a_ = std::get<0>(input);
  str_b_ = std::get<1>(input);
  global_result_ = 0;
  return true;
}

bool SizovDStringMismatchCountMPI::RunImpl() {
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized == 0 || finalized != 0) {
    return false;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(str_a_.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    global_result_ = 0;
    GetOutput() = 0;
    return true;
  }

  if (rank != 0) {
    str_a_.resize(static_cast<std::size_t>(total_size));
    str_b_.resize(static_cast<std::size_t>(total_size));
  }

  MPI_Bcast(str_a_.data(), total_size, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(str_b_.data(), total_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_result = 0;
  for (int i = rank; i < total_size; i += size) {
    if (str_a_[static_cast<std::size_t>(i)] != str_b_[static_cast<std::size_t>(i)]) {
      ++local_result;
    }
  }

  MPI_Reduce(&local_result, &global_result_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_result_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_result_;

  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_string_mismatch_count
