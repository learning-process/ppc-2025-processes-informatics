#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <iostream>

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
  return true;
}

bool SizovDStringMismatchCountMPI::RunImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cerr << "[Rank " << rank << "] Start RunImpl, size = " << size << "\n";

  const int total_size = static_cast<int>(str_a_.size());
  int local_result = 0;

  if (total_size > 0) {
    for (int i = rank; i < total_size; i += size) {
      if (str_a_[i] != str_b_[i]) {
        ++local_result;
      }
    }
  }

  int global_result = 0;

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;

  std::cerr << "[Rank " << rank << "] End RunImpl local = " << local_result << " global = " << global_result << "\n";

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return true;
}
}  // namespace sizov_d_string_mismatch_count
