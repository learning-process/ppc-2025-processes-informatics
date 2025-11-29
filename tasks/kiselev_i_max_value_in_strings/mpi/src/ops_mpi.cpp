#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_max_value_in_strings {

KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KiselevITestTaskMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }
  for (const auto &row : matrix) {
    if (row.empty()) {
      return false;
    }
  }
  return true;
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  const auto &matrix = GetInput();
  GetOutput().resize(matrix.size());
  return true;
}

bool KiselevITestTaskMPI::RunImpl() {
  const auto &matrix = GetInput();
  auto &result_vector = GetOutput();

  int world_rank{}, world_size{};
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // 1. Передаём количество строк
  int count_all_rows = 0;
  if (world_rank == 0) {
    count_all_rows = (int)matrix.size();
  }
  MPI_Bcast(&count_all_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);  // рассылка строк

  // 2. Сколько строк получает каждый процесс
  int count_of_row_per_process = count_all_rows / world_size;
  int reminder_row = count_all_rows % world_size;

  int local_row_count = count_of_row_per_process;
  if (world_rank < reminder_row) {
    local_row_count++;
  }

  // 3. Раздаём длины строк
  std::vector<int> all_row_lengths;  // только у root
  std::vector<int> local_row_lengths(local_row_count);
  std::vector<int> len_counts(world_size);  // строки по процессам
  std::vector<int> len_displs(world_size);  // сдвиги по строкам

  if (world_rank == 0) {
    all_row_lengths.resize(count_all_rows);
    for (int index = 0; index < count_all_rows; index++) {
      all_row_lengths[index] = matrix[index].size();
    }

    // формируем counts/displs по строкам
    int tmp_displs = 0;
    for (int p = 0; p < world_size; p++) {
      len_counts[p] = count_of_row_per_process;
      if (p < reminder_row) {
        len_counts[p]++;
      }

      len_displs[p] = tmp_displs;
      tmp_displs += len_counts[p];
    }
  }

  // может быть потребуется указать главный процесс для all_row_lenghts
  MPI_Scatterv(all_row_lengths.data(), len_counts.data(), len_displs.data(), MPI_INT, local_row_lengths.data(),
               local_row_count, MPI_INT, 0, MPI_COMM_WORLD);

  // 4. Раздаём сами элементы
  std::vector<int> val_counts(world_size), val_displs(world_size);

  if (world_rank == 0) {
    int tmp_offset = 0;
    for (int p = 0; p < world_size; p++) {
      int count_rows_for_process = len_counts[p];
      int start_row = len_displs[p];

      int tmp_count = 0;
      for (int index = 0; index < count_rows_for_process; index++) {
        tmp_count += all_row_lengths[start_row + index];
      }
      val_counts[p] = tmp_count;
      val_displs[p] = tmp_offset;
      tmp_offset += tmp_count;
    }
  }

  // рассылаем параметры раздачи
  MPI_Bcast(val_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(val_displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  // формируем `flat_matrix`
  std::vector<int> flat_matrix;
  if (world_rank == 0) {
    int count_elrms_on_flat_vector = val_displs[world_size - 1] + val_counts[world_size - 1];
    flat_matrix.reserve(count_elrms_on_flat_vector);

    for (int index = 0; index < count_all_rows; index++) {
      flat_matrix.insert(flat_matrix.end(), matrix[index].begin(), matrix[index].end());
    }
  }
  int my_elem_count = val_counts[world_rank];
  std::vector<int> local_values(my_elem_count);

  MPI_Scatterv(flat_matrix.data(), val_counts.data(), val_displs.data(), MPI_INT, local_values.data(), my_elem_count,
               MPI_INT, 0, MPI_COMM_WORLD);

  // 5. поиск local_res
  std::vector<int> local_result(local_row_count);

  for (int row = 0, pos = 0; row < local_row_count; row++) {
    int len = local_row_lengths[row];

    int tmp_max = local_values[pos];
    for (int j = 1; j < len; j++) {
      if (local_values[pos + j] > tmp_max) {
        tmp_max = local_values[pos + j];
      }
    }
    local_result[row] = tmp_max;
    pos += len;
  }

  // 6. Собираем данные
  if (world_rank == 0) {
    result_vector.resize(count_all_rows);
  }
  MPI_Gatherv(local_result.data(), local_row_count, MPI_INT, result_vector.data(), len_counts.data(), len_displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  // вроде не нужно, тк дальше ничего не нужно делать с данными
  // 7. Рассылаем всем итог
  if (count_all_rows > 0) {
    MPI_Bcast(result_vector.data(), count_all_rows, MPI_INT, 0, MPI_COMM_WORLD);
  }
  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_max_value_in_strings
