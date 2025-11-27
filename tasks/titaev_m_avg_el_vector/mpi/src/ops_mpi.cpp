#include "titaev_m_avg_el_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdint>

namespace titaev_m_avg_el_vector {

TitaevMAvgElVectorMPI::TitaevMAvgElVectorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TitaevMAvgElVectorMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
      return !GetInput().empty();
  }
  return true;
}

bool TitaevMAvgElVectorMPI::PreProcessingImpl() {
  return true;
}

bool TitaevMAvgElVectorMPI::RunImpl() {
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  unsigned int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<unsigned int>(GetInput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
      GetOutput() = 0.0;
      return true;
  }

  std::vector<int> send_counts(world_size);
  std::vector<int> displs(world_size);
  
  int remainder = total_size % world_size;
  int base_count = total_size / world_size;
  int offset = 0;

  for (int i = 0; i < world_size; ++i) {
    send_counts[i] = base_count + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += send_counts[i];
  }

  std::vector<int> local_vec(send_counts[rank]);

  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), send_counts.data(), displs.data(), MPI_INT,
                 local_vec.data(), send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, send_counts.data(), displs.data(), MPI_INT,
                 local_vec.data(), send_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);
  }
  int64_t local_sum = std::accumulate(local_vec.begin(), local_vec.end(), 0LL);
  int64_t global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = static_cast<double>(global_sum) / total_size;
  }

  return true;
}

bool TitaevMAvgElVectorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_avg_el_vector