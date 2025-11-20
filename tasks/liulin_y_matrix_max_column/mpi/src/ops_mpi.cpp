#include "liulin_y_matrix_max_column/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "liulin_y_matrix_max_column/common/include/common.hpp"
#include "util/include/util.hpp"

namespace liulin_y_matrix_max_column {

LiulinYMatrixMaxColumnMPI::LiulinYMatrixMaxColumnMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LiulinYMatrixMaxColumnMPI::ValidationImpl() {
  const auto& in = GetInput();

  if (in.empty() || in[0].empty())
    return false;

  size_t cols = in[0].size();
  for (const auto& row : in)
    if (row.size() != cols)
      return false;

  // Вектор выходных данных должен быть пустым
  return GetOutput().empty();
}

bool LiulinYMatrixMaxColumnMPI::PreProcessingImpl() {
  const size_t cols = GetInput()[0].size();
  GetOutput().assign(cols, std::numeric_limits<int>::min());
  return true;
}

bool LiulinYMatrixMaxColumnMPI::RunImpl() {
  const auto& in = GetInput();
  auto& out = GetOutput();

  const int rows = static_cast<int>(in.size());
  const int cols = static_cast<int>(in[0].size());

  int world_size = 1;
  int world_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // --- Распределяем строки по процессам ---
  int rows_per_rank = rows / world_size;
  int remainder = rows % world_size;

  int my_start = world_rank * rows_per_rank + std::min(world_rank, remainder);
  int my_rows  = rows_per_rank + (world_rank < remainder ? 1 : 0);
  int my_end   = my_start + my_rows;

  // Локальный максимум по каждому столбцу
  std::vector<int> local_max(cols, std::numeric_limits<int>::min());

  for (int r = my_start; r < my_end; r++) {
    for (int c = 0; c < cols; c++) {
      local_max[c] = std::max(local_max[c], in[r][c]);
    }
  }

  // Глобальный максимум — reduce по столбцам
  MPI_Reduce(local_max.data(), out.data(), cols, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool LiulinYMatrixMaxColumnMPI::PostProcessingImpl() {
  // Для MPI пост-обработка не нужна — возвращаем true
  return true;
}

}  // namespace liulin_y_matrix_max_column
