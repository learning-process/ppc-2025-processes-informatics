#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "egorova_l_find_max_val_col_matrix/common/include/common.hpp"
#include "util/include/util.hpp"

namespace egorova_l_find_max_val_col_matrix {

EgorovaLFindMaxValColMatrixMPI::EgorovaLFindMaxValColMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp = in;              // Создаем временную копию
  GetInput() = std::move(temp);  // Перемещаем
  GetOutput() = std::vector<int>(0);
}
bool EgorovaLFindMaxValColMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }

  size_t cols = GetInput()[0].size();
  for (const auto &row : GetInput()) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool EgorovaLFindMaxValColMatrixMPI::PreProcessingImpl() {
  return true;
}

bool EgorovaLFindMaxValColMatrixMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Определяем, пустая ли матрица
  int is_empty = 0;
  if (rank == 0) {
    is_empty = GetInput().empty() ? 1 : 0;
  }
  MPI_Bcast(&is_empty, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (is_empty) {
    GetOutput() = std::vector<int>();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  // Бродкастим размеры матрицы
  int rows = 0, cols = 0;
  if (rank == 0) {
    rows = static_cast<int>(GetInput().size());
    cols = static_cast<int>(GetInput()[0].size());
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Создаем и бродкастим плоскую матрицу
  std::vector<int> flat_matrix(rows * cols);
  if (rank == 0) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        flat_matrix[i * cols + j] = GetInput()[i][j];
      }
    }
  }
  MPI_Bcast(flat_matrix.data(), rows * cols, MPI_INT, 0, MPI_COMM_WORLD);

  // Распределяем столбцы между процессами
  int cols_per_proc = cols / size;
  int remainder = cols % size;

  // Определяем диапазон столбцов для текущего процесса
  int start_col, local_cols_count;
  if (rank < remainder) {
    start_col = rank * (cols_per_proc + 1);
    local_cols_count = cols_per_proc + 1;
  } else {
    start_col = remainder * (cols_per_proc + 1) + (rank - remainder) * cols_per_proc;
    local_cols_count = cols_per_proc;
  }

  // Каждый процесс вычисляет максимумы ТОЛЬКО для своих столбцов
  std::vector<int> local_max(local_cols_count, std::numeric_limits<int>::min());

  for (int local_idx = 0; local_idx < local_cols_count; ++local_idx) {
    int global_col = start_col + local_idx;
    for (int row = 0; row < rows; ++row) {
      int value = flat_matrix[row * cols + global_col];
      if (value > local_max[local_idx]) {
        local_max[local_idx] = value;
      }
    }
  }

  // Подготавливаем массивы для сбора результатов
  std::vector<int> all_max(cols, std::numeric_limits<int>::min());
  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);

  // Определяем сколько столбцов у каждого процесса
  for (int i = 0; i < size; ++i) {
    recv_counts[i] = (i < remainder) ? (cols_per_proc + 1) : cols_per_proc;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + recv_counts[i - 1];
  }

  // Собираем результаты от всех процессов
  MPI_Allgatherv(local_max.data(), local_cols_count, MPI_INT, all_max.data(), recv_counts.data(), displs.data(),
                 MPI_INT, MPI_COMM_WORLD);

  // Все процессы получают полный результат
  GetOutput() = all_max;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool EgorovaLFindMaxValColMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_find_max_val_col_matrix
