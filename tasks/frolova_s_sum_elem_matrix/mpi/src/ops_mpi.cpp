#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

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

  // Step 4: Prepare data for scattering
  std::vector<int> flat_data;
  std::vector<int> elem_counts(size, 0);
  std::vector<int> elem_displs(size, 0);

  if (rank == 0) {
    // Calculate total elements
    int64_t total_elements = 0;
    for (int i = 0; i < rows; i++) {
      total_elements += static_cast<int64_t>(row_sizes[i]);
    }
    //  Шаг 4.5: Рассылка elem_counts
    MPI_Bcast(elem_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    // Шаг 4.6: Рассылка elem_displs
    MPI_Bcast(elem_displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 5: Scatter element counts
    int my_elem_count = elem_counts[rank];

    // Шаг 6: Scatter matrix elements
    std::vector<int> local_data;
    if (my_elem_count > 0) {
      local_data.resize(my_elem_count);
    }

    // Создаем локальный указатель на данные отправки
    const int *send_data_ptr = nullptr;
    if (rank == 0) {
      send_data_ptr = flat_data.data();
    }

    // Передаем этот указатель
    MPI_Scatterv(const_cast<int *>(send_data_ptr), elem_counts.data(), elem_displs.data(), MPI_INT, local_data.data(),
                 my_elem_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 7: Compute local sum
    int64_t local_sum = 0;
    for (int i = 0; i < my_elem_count; i++) {
      local_sum += local_data[i];
    }

    // Step 8: Reduce to global sum
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

  bool FrolovaSSumElemMatrixMPI::PostProcessingImpl() {
    return true;
  }

}  // namespace frolova_s_sum_elem_matrix
