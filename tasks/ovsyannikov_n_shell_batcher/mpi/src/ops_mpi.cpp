#include "ovsyannikov_n_shell_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <stdexcept>
#include <vector>

namespace ovsyannikov_n_shell_batcher {

OvsyannikovNShellBatcherMPI::OvsyannikovNShellBatcherMPI(const InType &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
}

bool OvsyannikovNShellBatcherMPI::ValidationImpl() {
  return true;
}

bool OvsyannikovNShellBatcherMPI::PreProcessingImpl() {
  return true;
}

void OvsyannikovNShellBatcherMPI::ShellSort(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n < 2) {
    return;
  }

  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; i++) {
      int temp = arr[i];
      int j = i;
      while (j >= gap && arr[j - gap] > temp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = temp;
    }
  }
}

bool OvsyannikovNShellBatcherMPI::RunImpl() {
  int mpi_flag = 0;
  MPI_Initialized(&mpi_flag);
  if (!mpi_flag) {
    MPI_Init(nullptr, nullptr);
    need_finalize_ = true;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_len = 0;
  if (rank == 0) {
    total_len = static_cast<int>(this->GetInput().size());
  }
  MPI_Bcast(&total_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_len == 0) {
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  int chunk = total_len / size;
  int rem = total_len % size;
  for (int i = 0; i < size; i++) {
    counts[i] = chunk + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }

  std::vector<int> local_data(counts[rank]);
  MPI_Scatterv(rank == 0 ? this->GetInput().data() : nullptr, counts.data(), displs.data(), MPI_INT, local_data.data(),
               counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  ShellSort(local_data);

  if (rank == 0) {
    this->GetOutput().resize(total_len);
  }

  MPI_Gatherv(local_data.data(), counts[rank], MPI_INT, rank == 0 ? this->GetOutput().data() : nullptr, counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    ShellSort(this->GetOutput());
  }

  if (rank != 0) {
    this->GetOutput().resize(total_len);
  }

  MPI_Bcast(this->GetOutput().data(), total_len, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool OvsyannikovNShellBatcherMPI::PostProcessingImpl() {
  if (need_finalize_) {
    MPI_Finalize();
    need_finalize_ = false;
  }
  return true;
}

}  // namespace ovsyannikov_n_shell_batcher
