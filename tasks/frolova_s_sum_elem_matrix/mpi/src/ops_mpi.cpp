#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

namespace frolova_s_sum_elem_matrix {

FrolovaSSumElemMatrixMPI::FrolovaSSumElemMatrixMPI(const InType &) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool FrolovaSSumElemMatrixMPI::PreProcessingImpl() {
  return true;
}

bool FrolovaSSumElemMatrixMPI::ValidationImpl() {
  return true;
}

bool FrolovaSSumElemMatrixMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cerr << "[Rank " << rank << "] RunImpl started\n";

  const auto &matrix = GetInput();
  int rows = (rank == 0) ? static_cast<int>(matrix.size()) : 0;

  std::cerr << "[Rank " << rank << "] Before Bcast, rows=" << rows << "\n";
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::cerr << "[Rank " << rank << "] After Bcast, rows=" << rows << "\n";

  if (rows == 0) {
    if (rank == 0) {
      GetOutput() = static_cast<OutType>(0);
    }
    std::cerr << "[Rank " << rank << "] Matrix is empty, exiting\n";
    return true;
  }

  // Размеры строк
  std::vector<int> row_sizes(rows);
  if (rank == 0) {
    for (int i = 0; i < rows; ++i) {
      row_sizes[i] = static_cast<int>(matrix[i].size());
    }
  }
  MPI_Bcast(row_sizes.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);
  std::cerr << "[Rank " << rank << "] Row sizes broadcasted\n";

  // Распределение строк
  std::vector<int> row_counts(size, 0), row_displs(size, 0);
  int base = rows / size, rem = rows % size, offset = 0;
  for (int i = 0; i < size; ++i) {
    row_counts[i] = base + (i < rem ? 1 : 0);
    row_displs[i] = offset;
    offset += row_counts[i];
  }
  std::cerr << "[Rank " << rank << "] Row distribution: start=" << row_displs[rank] << ", count=" << row_counts[rank]
            << "\n";

  int my_start = row_displs[rank];
  int my_rows = row_counts[rank];

  int64_t local_sum = 0;

  if (rank == 0) {
    std::cerr << "[Rank 0] Calculating sum for own rows\n";
    for (int i = my_start; i < my_start + my_rows; ++i) {
      for (int val : matrix[i]) {
        local_sum += val;
      }
    }

    // Отправляем строки остальным процессам
    for (int p = 1; p < size; ++p) {
      int start = row_displs[p];
      int count = row_counts[p];
      for (int i = 0; i < count; ++i) {
        MPI_Send(matrix[start + i].data(), row_sizes[start + i], MPI_INT, p, 0, MPI_COMM_WORLD);
        std::cerr << "[Rank 0] Sent row " << start + i << " to rank " << p << "\n";
      }
    }
  } else {
    std::cerr << "[Rank " << rank << "] Receiving rows\n";
    for (int i = 0; i < my_rows; ++i) {
      std::vector<int> row(row_sizes[my_start + i]);
      MPI_Recv(row.data(), row.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::cerr << "[Rank " << rank << "] Received row " << my_start + i << "\n";
      for (int val : row) {
        local_sum += val;
      }
    }
  }

  std::cerr << "[Rank " << rank << "] Local sum = " << local_sum << "\n";

  int64_t global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = static_cast<OutType>(global_sum);
    std::cerr << "[Rank 0] Global sum = " << global_sum << "\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  std::cerr << "[Rank " << rank << "] RunImpl finished\n";

  return true;
}

bool FrolovaSSumElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_sum_elem_matrix
