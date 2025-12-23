#include "ovsyannikov_n_shell_batcher/seq/include/ops_seq.hpp"
#include <mpi.h>
#include <algorithm>
#include <vector>

namespace ovsyannikov_n_shell_batcher {

OvsyannikovNShellBatcherSEQ::OvsyannikovNShellBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool OvsyannikovNShellBatcherSEQ::ValidationImpl() {
  return true;
}

bool OvsyannikovNShellBatcherSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool OvsyannikovNShellBatcherSEQ::RunImpl() {
  int rank = 0;
  int is_mpi_init = 0;
  MPI_Initialized(&is_mpi_init);
  if (is_mpi_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Считает только Rank 0
  if (rank == 0) {
    auto &arr = GetOutput();
    int n = static_cast<int>(arr.size());
    if (n > 1) {
      for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; i++) {
          int temp = arr[i];
          int j;
          for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
            arr[j] = arr[j - gap];
          }
          arr[j] = temp;
        }
      }
    }
  }

  // Рассылаем всем, если мы в MPI среде
  if (is_mpi_init) {
    int n = 0;
    if (rank == 0) n = static_cast<int>(GetOutput().size());
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) GetOutput().resize(n);
    if (n > 0) {
      MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }
  }

  return true;
}

bool OvsyannikovNShellBatcherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_shell_batcher