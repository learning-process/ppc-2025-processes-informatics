#include "frolova_s_sum_elem_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "frolova_s_sum_elem_matrix/common/include/common.hpp"

namespace frolova_s_sum_elem_matrix {

FrolovaSSumElemMatrixMPI::FrolovaSSumElemMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = 0;
}

bool FrolovaSSumElemMatrixMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool FrolovaSSumElemMatrixMPI::PostProcessingImpl() {
  return true;
}

bool FrolovaSSumElemMatrixMPI::ValidationImpl() {
  return true;
}

bool FrolovaSSumElemMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Входящая матрица доступна только на Ранге 0
  const auto &matrix = GetInput();

  // Step 1: Get input and broadcast rows count
  int rows = 0;
  if (rank == 0) {
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

  // Step 2: Broadcast row sizes (Обязательно для рваной матрицы)
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

  // Step 4: Prepare data for scattering and calculate element parameters
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

    // 4.3. Calculate correct elem_counts and elem_displs (Параметры Scatterv)
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

    // 4.4. Fill flat_data (Сглаживание матрицы)
    int idx = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < row_sizes[i]; j++) {
        flat_data[idx++] = matrix[i][j];
      }
    }
  }  // Конец блока rank == 0

  // Шаг 4.5: Рассылка elem_counts
  MPI_Bcast(elem_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  // Шаг 4.6: Рассылка elem_displs
  MPI_Bcast(elem_displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  // Step 5: Get local element count
  int my_elem_count = elem_counts[rank];

  // Шаг 6: Scatter matrix elements
  std::vector<int> local_data;
  if (my_elem_count > 0) {
    local_data.resize(my_elem_count);
  }

  // Используем тернарный оператор для sendbuf (ранг 0 отправляет данные, остальные отправляют nullptr)
  MPI_Scatterv(rank == 0 ? flat_data.data() : nullptr, elem_counts.data(), elem_displs.data(), MPI_INT,
               local_data.data(), my_elem_count, MPI_INT, 0, MPI_COMM_WORLD);

  // Step 7: Compute local sum
  int64_t local_sum = 0;
  for (int i = 0; i < my_elem_count; i++) {
    local_sum += local_data[i];
  }

  // Step 8: Reduce to global sum (Используем MPI_LONG_LONG для int64_t)
  int64_t global_sum = 0;
  // Результат сохраняется только на Ранге 0
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
