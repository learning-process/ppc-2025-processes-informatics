#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"
#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp

namespace frolova_s_sum_elem_matrix {

void FrolovaSSumElemMatrixMPI::BroadcastMetadata(int &rows) {
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void FrolovaSSumElemMatrixMPI::BroadcastRowSizes(int rows, std::vector<int> &row_sizes) {
  MPI_Bcast(row_sizes.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);
}

void FrolovaSSumElemMatrixMPI::FlattenMatrixOnRoot(int rank, const std::vector<std::vector<int>> &matrix,
                                                   const std::vector<int> &row_sizes, std::vector<int> &flat_data,
                                                   std::vector<int> &displs) {
  if (rank == 0) {
    int total_elements = 0;
    for (size_t i = 0; i < row_sizes.size(); i++) {
      displs[i] = total_elements;
      total_elements += row_sizes[i];
    }

    flat_data.resize(total_elements);
    int idx = 0;
    for (size_t i = 0; i < matrix.size(); i++) {
      for (int j = 0; j < row_sizes[i]; j++) {
        flat_data[idx++] = matrix[i][j];
      }
    }
  }
}

void FrolovaSSumElemMatrixMPI::ComputeDistribution(int size, int rows, std::vector<int> &counts,
                                                   std::vector<int> &displacements) {
  int total_rows_per_process = rows / size;
  int remainder = rows % size;

  for (int i = 0; i < size; i++) {
    counts[i] = total_rows_per_process + (i < remainder ? 1 : 0);
    if (i > 0) {
      displacements[i] = displacements[i - 1] + counts[i - 1];
    }
  }
}

void FrolovaSSumElemMatrixMPI::ComputeSendCounts(int rank, int size, const std::vector<int> &counts,
                                                 const std::vector<int> &displacements,
                                                 const std::vector<int> &row_sizes, std::vector<int> &sendcounts,
                                                 std::vector<int> &senddispls) {
  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      senddispls[i] = (i == 0) ? 0 : senddispls[i - 1] + sendcounts[i - 1];
      for (int j = 0; j < counts[i]; j++) {
        int row_idx = displacements[i] + j;
        sendcounts[i] += row_sizes[row_idx];
      }
    }
  }
}

FrolovaSSumElemMatrixMPI::FrolovaSSumElemMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool FrolovaSSumElemMatrixMPI::ValidationImpl() {
  return true;
}

bool FrolovaSSumElemMatrixMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool FrolovaSSumElemMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Step 1: Get input and broadcast rows count
  std::vector<std::vector<int>> matrix;
  int rows = 0;
  if (rank == 0) {
    matrix = GetInput();
    rows = static_cast<int>(matrix.size());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Handle empty matrix
  if (rows == 0) {
    if (rank == 0) {
      GetOutput() = static_cast<OutType>(0);
    }
    return true;
  }

  // Step 2: Broadcast row sizes
  std::vector<int> row_sizes(rows);
  if (rank == 0) {
    for (int i = 0; i < rows; i++) {
      row_sizes[i] = static_cast<int>(matrix[i].size());
    }
  }
  MPI_Bcast(row_sizes.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);

  // Step 3: Calculate row distribution
  std::vector<int> row_counts(size, 0);
  std::vector<int> row_displs(size, 0);
  int base_rows = rows / size;
  int remainder = rows % size;
  int current_displ = 0;

  for (int i = 0; i < size; i++) {
    row_counts[i] = base_rows + (i < remainder ? 1 : 0);
    row_displs[i] = current_displ;
    current_displ += row_counts[i];
  }

  // Step 4: Prepare data for scattering and calculate parameters
  std::vector<int> flat_data;
  std::vector<int> elem_counts(size, 0);
  std::vector<int> elem_displs(size, 0);

  if (rank == 0) {
    // 4.1. Calculate total elements (safe int64_t)
    int64_t total_elements = 0;
    for (int i = 0; i < rows; i++) {
      total_elements += static_cast<int64_t>(row_sizes[i]);
    }
    // 4.2. Resize flat_data (safe static_cast)
    flat_data.resize(static_cast<size_t>(total_elements));

    // 4.3. Calculate correct elem_counts and elem_displs
    int current_elem_displ = 0;
    for (int proc = 0; proc < size; proc++) {
      elem_counts[proc] = 0;
      for (int j = 0; j < row_counts[proc]; j++) {
        int row_idx = row_displs[proc] + j;
        elem_counts[proc] += row_sizes[row_idx];
      }
      elem_displs[proc] = current_elem_displ;
      current_elem_displ += elem_counts[proc];
    }

    // 4.4. Fill flat_data
    int idx = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < row_sizes[i]; j++) {
        flat_data[idx++] = matrix[i][j];
      }
    }

    // --- ОТЛАДКА 1: Параметры Scatterv на Ранге 0 ---
    std::cerr << "[ RANK " << rank << " ] DEBUG: Parameters for Scatterv calculated.\n";
    std::cerr << "[ RANK " << rank << " ] DEBUG: elem_counts: ";
    for (int count : elem_counts) {
      std::cerr << count << " ";
    }
    std::cerr << "\n[ RANK " << rank << " ] DEBUG: elem_displs: ";
    for (int displ : elem_displs) {
      std::cerr << displ << " ";
    }
    std::cerr << "\n";
  }

  // Шаг 4.5: Рассылка elem_counts (вызывается всеми)
  MPI_Bcast(elem_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  // Шаг 4.6: Рассылка elem_displs (вызывается всеми)
  MPI_Bcast(elem_displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  // Step 5: Get local element count
  int my_elem_count = elem_counts[rank];

  // --- ОТЛАДКА 2: my_elem_count на каждом Ранге ---
  std::cerr << "[ RANK " << rank << " ] DEBUG: My element count (my_elem_count): " << my_elem_count << "\n";

  // Шаг 6: Scatter matrix elements
  std::vector<int> local_data;
  if (my_elem_count > 0) {
    local_data.resize(my_elem_count);
  }

  // Используем тернарный оператор для sendbuf, чтобы избежать const_cast/временных указателей.
  MPI_Scatterv(rank == 0 ? flat_data.data() : nullptr, elem_counts.data(), elem_displs.data(), MPI_INT,
               local_data.data(), my_elem_count, MPI_INT, 0, MPI_COMM_WORLD);

  // --- ОТЛАДКА 3: Полученные данные на Ранге 1 (или других рабочих рангах) ---
  if (rank != 0 && my_elem_count > 0) {
    // Печатаем первый и последний элементы для проверки корректности данных
    std::cerr << "[ RANK " << rank << " ] DEBUG: Received data sample: First=" << local_data[0]
              << ", Last=" << local_data[my_elem_count - 1] << "\n";
  }

  // Step 7: Compute local sum
  int64_t local_sum = 0;
  for (int i = 0; i < my_elem_count; i++) {
    local_sum += local_data[i];
  }

  // --- ОТЛАДКА 4: Local Sum на каждом Ранге ---
  std::cerr << "[ RANK " << rank << " ] DEBUG: Local sum calculated: " << local_sum << "\n";

  // Step 8: Reduce to global sum (Используем MPI_LONG_LONG для int64_t)
  int64_t global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  // Step 9: Set output ONLY on rank 0
  if (rank == 0) {
    GetOutput() = static_cast<OutType>(global_sum);
  }

  // Синхронизируем все процессы перед выходом
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

}  // namespace frolova_s_sum_elem_matrix
