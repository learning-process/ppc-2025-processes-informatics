#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"

namespace sizov_d_bubble_sort {

SizovDBubbleSortMPI::SizovDBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortMPI::ValidationImpl() {
  const auto &input = GetInput();
  return !input.empty();
}

bool SizovDBubbleSortMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_size = static_cast<int>(data_.size());
  const int base = total_size / size;
  const int remainder = total_size % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  int offset = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<int> local_data(counts[rank]);
  MPI_Scatterv(data_.data(), counts.data(), displs.data(), MPI_INT, local_data.data(), counts[rank], MPI_INT, 0,
               MPI_COMM_WORLD);

  std::sort(local_data.begin(), local_data.end());

  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<int> global_data;
  if (rank == 0) {
    global_data.resize(total_size);
  }

  MPI_Gatherv(local_data.data(), counts[rank], MPI_INT, global_data.data(), counts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    std::sort(global_data.begin(), global_data.end());
    GetOutput() = global_data;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool SizovDBubbleSortMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n = (rank == 0) ? static_cast<int>(GetOutput().size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().assign(n, 0);
  }

  if (n > 0) {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace sizov_d_bubble_sort
