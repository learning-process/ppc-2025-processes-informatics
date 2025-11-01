/*#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <iostream>
#include <vector>

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
    std::cerr << "[SizovDStringMismatchCountMPI] MPI is not initialized\n";
    return false;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_size = static_cast<int>(str_a_.size());
  if (total_size == 0) {
    int zero = 0;
    MPI_Bcast(&zero, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = 0;
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    const int base_chunk = total_size / size;
    const int remainder = total_size % size;

    for (int i = 0; i < size; ++i) {
      counts[i] = base_chunk;
      if (i < remainder) {
        counts[i] += 1;
      }
    }

    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  MPI_Bcast(counts.data(), static_cast<int>(counts.size()), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), static_cast<int>(displs.size()), MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<char> local_a(counts[rank]);
  std::vector<char> local_b(counts[rank]);

  MPI_Scatterv(str_a_.data(), counts.data(), displs.data(), MPI_CHAR, local_a.data(), counts[rank], MPI_CHAR, 0,
               MPI_COMM_WORLD);

  MPI_Scatterv(str_b_.data(), counts.data(), displs.data(), MPI_CHAR, local_b.data(), counts[rank], MPI_CHAR, 0,
               MPI_COMM_WORLD);

  int local_result = 0;
  for (int i = 0; i < counts[rank]; ++i) {
    if (local_a[i] != local_b[i]) {
      ++local_result;
    }
  }

  int global_result = 0;
  MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;

  std::cerr << "[Rank " << rank << "] local=" << local_result << ", chunk=" << counts[rank]
            << ", global=" << global_result << "\n";

  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_string_mismatch_count*/
