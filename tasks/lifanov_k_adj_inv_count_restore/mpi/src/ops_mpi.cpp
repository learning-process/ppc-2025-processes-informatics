#include "lifanov_k_adj_inv_count_restore/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
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
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t total_pairs = n - 1;
  const std::size_t base = total_pairs / static_cast<std::size_t>(size);
  const std::size_t rem = total_pairs % static_cast<std::size_t>(size);

  const std::size_t start_pair = (static_cast<std::size_t>(rank) * base) + std::min<std::size_t>(rank, rem);
  const std::size_t count = base + (std::cmp_less(rank, static_cast<int>(rem)) ? 1 : 0);
  const std::size_t end_pair = start_pair + count;

  int local = 0;
  for (std::size_t i = start_pair; i < end_pair; ++i) {
    if (data[i] > data[i + 1]) {
      local++;
    }
  }

  int global = 0;
  MPI_Reduce(&local, &global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = global;

  return true;
}

bool LifanovKAdjacentInversionCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_adj_inv_count_restore
