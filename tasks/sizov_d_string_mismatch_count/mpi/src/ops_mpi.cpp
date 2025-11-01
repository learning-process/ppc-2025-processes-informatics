#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sizov_d_string_mismatch_count {

SizovDStringMismatchCountMPI::SizovDStringMismatchCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SizovDStringMismatchCountMPI::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return !a.empty() && a.size() == b.size();
}

bool SizovDStringMismatchCountMPI::PreProcessingImpl() {
  const auto &[a, b] = GetInput();
  str_a_ = a;
  str_b_ = b;
  GetOutput() = 0;
  return !str_a_.empty() && str_a_.size() == str_b_.size();
}

bool SizovDStringMismatchCountMPI::RunImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    std::cerr << "[task] MPI not initialized\n";
    return false;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_size = static_cast<int>(str_a_.size());
  if (total_size == 0) {
    GetOutput() = 0;
    return true;
  }

  std::vector<int> counts(size, 0);
  std::vector<int> displs(size, 0);

  if (rank == 0) {
    const int base = total_size / size;
    const int rem = total_size % size;
    for (int i = 0; i < size; ++i) {
      counts[i] = base + (i < rem ? 1 : 0);
    }
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

  int local_diff = 0;
  for (int i = 0; i < counts[rank]; ++i) {
    if (local_a[i] != local_b[i]) {
      ++local_diff;
    }
  }

  int global_diff = 0;
  MPI_Reduce(&local_diff, &global_diff, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_diff, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  GetOutput() = global_diff;

  std::cerr << "[Rank " << rank << "] local=" << local_diff << " chunk=" << counts[rank] << " total=" << total_size
            << " global=" << global_diff << "\n";

  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace sizov_d_string_mismatch_count
