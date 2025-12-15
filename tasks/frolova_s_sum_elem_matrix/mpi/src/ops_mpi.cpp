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
  std::cerr << "RunImpl entered (before MPI init)\n";

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Входящая матрица доступна только на rank 0
  const int *data = nullptr;
  int rows = 0, cols = 0;

  if (rank == 0) {
    data = reinterpret_cast<const int *>(taskData->inputs[0]);
    rows = *reinterpret_cast<const int *>(taskData->inputs[1]);
    cols = *reinterpret_cast<const int *>(taskData->inputs[2]);
  }

  // Рассылаем количество строк всем процессам
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0) {
    if (rank == 0) {
      *reinterpret_cast<OutType *>(taskData->outputs[0]) = static_cast<OutType>(0);
    }
    return true;
  }

  // Создаём локальную рваную матрицу на rank 0
  std::vector<std::vector<int>> matrix;
  if (rank == 0) {
    matrix.resize(rows);
    for (int i = 0; i < rows; ++i) {
      matrix[i].resize(cols);
      for (int j = 0; j < cols; ++j) {
        matrix[i][j] = data[i * cols + j];
      }
    }
  }

  // Размеры строк
  std::vector<int> row_sizes(rows);
  if (rank == 0) {
    for (int i = 0; i < rows; ++i) {
      row_sizes[i] = static_cast<int>(matrix[i].size());
    }
  }
  MPI_Bcast(row_sizes.data(), rows, MPI_INT, 0, MPI_COMM_WORLD);

  // Распределение строк по процессам
  std::vector<int> row_counts(size, 0), row_displs(size, 0);
  int base_rows = rows / size;
  int remainder = rows % size;
  int current_displ = 0;
  for (int i = 0; i < size; ++i) {
    row_counts[i] = base_rows + (i < remainder ? 1 : 0);
    row_displs[i] = current_displ;
    current_displ += row_counts[i];
  }

  // Подготовка сглаженной матрицы для Scatterv
  std::vector<int> flat_data;
  std::vector<int> elem_counts(size, 0), elem_displs(size, 0);

  if (rank == 0) {
    int64_t total_elements = 0;
    for (int i = 0; i < rows; ++i) {
      total_elements += static_cast<int64_t>(row_sizes[i]);
    }
    flat_data.resize(static_cast<size_t>(total_elements));

    int current_elem_displ = 0;
    for (int proc = 0; proc < size; ++proc) {
      elem_counts[proc] = 0;
      for (int j = 0; j < row_counts[proc]; ++j) {
        int row_idx = row_displs[proc] + j;
        elem_counts[proc] += row_sizes[row_idx];
      }
      elem_displs[proc] = current_elem_displ;
      current_elem_displ += elem_counts[proc];
    }

    // Сглаживаем матрицу
    int idx = 0;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < row_sizes[i]; ++j) {
        flat_data[idx++] = matrix[i][j];
      }
    }
  }

  // Рассылаем параметры Scatterv всем процессам
  MPI_Bcast(elem_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(elem_displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  int my_elem_count = elem_counts[rank];
  std::vector<int> local_data(my_elem_count);

  MPI_Scatterv(rank == 0 ? flat_data.data() : nullptr, elem_counts.data(), elem_displs.data(), MPI_INT,
               local_data.data(), my_elem_count, MPI_INT, 0, MPI_COMM_WORLD);

  // Вычисляем локальную сумму
  int64_t local_sum = 0;
  for (int i = 0; i < my_elem_count; ++i) {
    local_sum += local_data[i];
  }

  // Сводим локальные суммы в глобальную
  int64_t global_sum = 0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    *reinterpret_cast<OutType *>(taskData->outputs[0]) = static_cast<OutType>(global_sum);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool FrolovaSSumElemMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_sum_elem_matrix
