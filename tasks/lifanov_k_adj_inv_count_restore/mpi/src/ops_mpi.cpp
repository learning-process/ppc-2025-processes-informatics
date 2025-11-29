#include "lifanov_k_adj_inv_count_restore/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "lifanov_k_adj_inv_count_restore/common/include/common.hpp"

namespace lifanov_k_adj_inv_count_restore {

LifanovKAdjacentInversionCountMPI::LifanovKAdjacentInversionCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool LifanovKAdjacentInversionCountMPI::ValidationImpl() {
  return !GetInput().empty() && (GetOutput() == 0);
}

bool LifanovKAdjacentInversionCountMPI::PreProcessingImpl() {
  return true;
}

bool LifanovKAdjacentInversionCountMPI::RunImpl() {
  const auto &data = GetInput();
  const std::size_t n = data.size();

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (n < 2) {
    GetOutput() = 0;
    return true;
  }

  const std::size_t total_pairs = n - 1;
  const std::size_t base = total_pairs / static_cast<std::size_t>(world_size);
  const std::size_t rem = total_pairs % static_cast<std::size_t>(world_size);

  std::vector<int> sendcounts(world_size, 0);
  std::vector<int> displs(world_size, 0);

  // Формирование sendcounts
  for (int proc_idx = 0; proc_idx < world_size; ++proc_idx) {
    const bool extra_pair = static_cast<std::size_t>(proc_idx) < rem;
    const std::size_t local_pairs = base + (extra_pair ? 1 : 0);
    sendcounts[proc_idx] = static_cast<int>(local_pairs + 1);
  }

  // Формирование displs с перекрытием на один элемент
  for (int proc_idx = 1; proc_idx < world_size; ++proc_idx) {
    displs[proc_idx] = displs[proc_idx - 1] + sendcounts[proc_idx - 1] - 1;
  }

  int local_size = sendcounts[rank];
  std::vector<int> local(local_size);

  MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_INT, local.data(), local_size, MPI_INT, 0,
               MPI_COMM_WORLD);

  int local_inv = 0;
  for (int i = 0; i + 1 < local_size; ++i) {
    if (local[i] > local[i + 1]) {
      local_inv++;
    }
  }

  int global = 0;
  MPI_Reduce(&local_inv, &global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = global;

  return true;
}

bool LifanovKAdjacentInversionCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_adj_inv_count_restore
