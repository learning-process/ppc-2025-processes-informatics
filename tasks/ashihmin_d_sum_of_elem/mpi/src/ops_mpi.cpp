#include "ashihmin_d_sum_of_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

namespace ashihmin_d_sum_of_elem {

AshihminDElemVecsSumMPI::AshihminDElemVecsSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AshihminDElemVecsSumMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool AshihminDElemVecsSumMPI::PreProcessingImpl() {
  return true;
}

bool AshihminDElemVecsSumMPI::RunImpl() {
  const auto &vec = GetInput();

  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  size_t n = vec.size();
  for (size_t i = 0; i < size; ++i) {
    counts[i] = n / static_cast<int>(size);
    if (i < static_cast<size_t>(n % size)) {
      counts[i]++;
    }
  }
  std::vector<int> local(counts[rank]);
  MPI_Scatterv(vec.data(), counts.data(), displs.data(), MPI_INT, local.data(), counts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  OutType local_sum = 0;
  for (int x : local) {
    local_sum += x;
  }

  OutType global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;
  return true;
}

bool AshihminDElemVecsSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace ashihmin_d_sum_of_elem
