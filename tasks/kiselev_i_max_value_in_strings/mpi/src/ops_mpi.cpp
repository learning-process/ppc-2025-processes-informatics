// #include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"

// #include <mpi.h>

// #include <utility>
// #include <vector>

// #include "kiselev_i_max_value_in_strings/common/include/common.hpp"

// namespace kiselev_i_max_value_in_strings {

// KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
//   SetTypeOfTask(GetStaticTypeOfTask());
//   auto in_copy = in;
//   GetInput() = std::move(in_copy);
//   GetOutput().clear();
// }

// bool KiselevITestTaskMPI::ValidationImpl() {
//   const auto &matrix = GetInput();
//   if (matrix.empty()) {
//     return false;
//   }
//   for (const auto &row : matrix) {
//     if (row.empty()) {
//       return false;
//     }
//   }
//   return true;
// }

// bool KiselevITestTaskMPI::PreProcessingImpl() {
//   const auto &matrix = GetInput();
//   GetOutput().resize(matrix.size());
//   return true;
// }

// bool KiselevITestTaskMPI::RunImpl() {
//   const auto &matrix = GetInput();
//   auto &result_vector = GetOutput();

//   int world_rank{0}, world_size{0};
//   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//   // 1. Передаём количество строк
//   int count_all_rows = static_cast<int>(matrix.size());
//   MPI_Bcast(&count_all_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);  // рассылка строк

//   // 2. Сколько строк получает каждый процесс
//   int count_of_row_per_process = count_all_rows / world_size;
//   int reminder_row = count_all_rows % world_size;
//   int local_row_count = count_of_row_per_process;
//   if (world_rank < reminder_row) {
//     local_row_count++;
//   }

//   // 3. Раздаём длины строк
//   std::vector<int> all_row_lengths;  // only root
//   std::vector<int> local_row_lengths(local_row_count);
//   std::vector<int> len_counts(world_size);  // rows by processes
//   std::vector<int> len_displs(world_size);  // displs by rows

//   if (world_rank == 0) {
//     all_row_lengths.resize(static_cast<size_t>(count_all_rows));
//     for (int index = 0; index < count_all_rows; index++) {
//       all_row_lengths[static_cast<size_t>(index)] = static_cast<int>(matrix[static_cast<size_t>(index)].size());
//     }

//     // формируем counts/displs по строкам
//     int tmp_displs = 0;
//     for (int p = 0; p < world_size; p++) {
//       len_counts[p] = count_of_row_per_process;
//       if (p < reminder_row) {
//         len_counts[p]++;
//       }

//       len_displs[p] = tmp_displs;
//       tmp_displs += len_counts[p];
//     }
//   }

//   // может быть потребуется указать главный процесс для all_row_lenghts
//   MPI_Scatterv(all_row_lengths.data(), len_counts.data(), len_displs.data(), MPI_INT, local_row_lengths.data(),
//                local_row_count, MPI_INT, 0, MPI_COMM_WORLD);

//   // 4. Раздаём сами элементы
//   std::vector<int> val_counts(world_size), val_displs(world_size);

//   if (world_rank == 0) {
//     int tmp_offset = 0;
//     for (int p = 0; p < world_size; p++) {
//       int count_rows_for_process = len_counts[p];
//       int start_row = len_displs[p];

//       int tmp_count = 0;
//       for (int index = 0; index < count_rows_for_process; index++) {
//         tmp_count += all_row_lengths[start_row + index];
//       }
//       val_counts[p] = tmp_count;
//       val_displs[p] = tmp_offset;
//       tmp_offset += tmp_count;
//     }
//   }

//   // рассылаем параметры раздачи
//   MPI_Bcast(val_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
//   MPI_Bcast(val_displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

//   // формируем `flat_matrix`
//   std::vector<int> flat_matrix;
//   if (world_rank == 0) {
//     int count_elrms_on_flat_vector = val_displs[world_size - 1] + val_counts[world_size - 1];
//     flat_matrix.reserve(count_elrms_on_flat_vector);

//     for (int index = 0; index < count_all_rows; index++) {
//       flat_matrix.insert(flat_matrix.end(), matrix[index].begin(), matrix[index].end());
//     }
//   }
//   int my_elem_count = val_counts[world_rank];
//   std::vector<int> local_values(my_elem_count);

//   MPI_Scatterv(flat_matrix.data(), val_counts.data(), val_displs.data(), MPI_INT, local_values.data(), my_elem_count,
//                MPI_INT, 0, MPI_COMM_WORLD);

//   // 5. поиск local_res
//   std::vector<int> local_result(local_row_count);

//   for (int row = 0, pos = 0; row < local_row_count; row++) {
//     int len = local_row_lengths[row];

//     int tmp_max = local_values[pos];
//     for (int j = 1; j < len; j++) {
//       if (local_values[pos + j] > tmp_max) {
//         tmp_max = local_values[pos + j];
//       }
//     }
//     local_result[row] = tmp_max;
//     pos += len;
//   }

//   // 6. Собираем данные
//   if (world_rank == 0) {
//     result_vector.resize(count_all_rows);
//   }
//   MPI_Gatherv(local_result.data(), local_row_count, MPI_INT, result_vector.data(), len_counts.data(),
//   len_displs.data(),
//               MPI_INT, 0, MPI_COMM_WORLD);

//   // вроде не нужно, тк дальше ничего не нужно делать с данными
//   // 7. Рассылаем всем итог
//   if (count_all_rows > 0) {
//     MPI_Bcast(result_vector.data(), count_all_rows, MPI_INT, 0, MPI_COMM_WORLD);
//   }
//   return true;
// }

// bool KiselevITestTaskMPI::PostProcessingImpl() {
//   return true;
// }

// }  // namespace kiselev_i_max_value_in_strings
#include "kiselev_i_max_value_in_strings/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "kiselev_i_max_value_in_strings/common/include/common.hpp"

namespace kiselev_i_max_value_in_strings {

KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
  GetOutput().clear();
}

bool KiselevITestTaskMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }
  return std::all_of(matrix.begin(), matrix.end(), [](const auto &row) { return !row.empty(); });
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  const auto &matrix = GetInput();
  GetOutput().resize(matrix.size());
  return true;
}

// --- Вспомогательные функции для RunImpl ---
int KiselevITestTaskMPI::CalculateLocalRowCount(int total_rows, int world_rank, int world_size) {
  int count_of_row_per_process = total_rows / world_size;
  int reminder_row = total_rows % world_size;
  int local_row_count = count_of_row_per_process;
  if (world_rank < reminder_row) {
    local_row_count++;
  }
  return local_row_count;
}

void KiselevITestTaskMPI::DistributeRowLengths(const std::vector<std::vector<int>> &matrix, int total_rows,
                                               int world_rank, int world_size, std::vector<int> &local_row_lengths,
                                               std::vector<int> &len_counts, std::vector<int> &len_displs) {
  std::vector<int> all_row_lengths;
  int count_of_row_per_process = total_rows / world_size;
  int reminder_row = total_rows % world_size;

  if (world_rank == 0) {
    all_row_lengths.resize(static_cast<size_t>(total_rows));
    for (int index = 0; index < total_rows; ++index) {
      all_row_lengths[static_cast<size_t>(index)] = static_cast<int>(matrix[static_cast<size_t>(index)].size());
    }

    int tmp_displs = 0;
    for (int process = 0; process < world_size; ++process) {
      len_counts[process] = count_of_row_per_process;
      if (process < reminder_row) {
        len_counts[process]++;
      }
      len_displs[process] = tmp_displs;
      tmp_displs += len_counts[process];
    }
  }

  MPI_Scatterv(all_row_lengths.data(), len_counts.data(), len_displs.data(), MPI_INT, local_row_lengths.data(),
               static_cast<int>(local_row_lengths.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

void KiselevITestTaskMPI::DistributeValues(const std::vector<std::vector<int>> &matrix, int world_rank, int world_size,
                                           const std::vector<int> &len_counts, const std::vector<int> &len_displs,
                                           std::vector<int> &local_values) {
  std::vector<int> val_counts(world_size);
  std::vector<int> val_displs(world_size);

  if (world_rank == 0) {
    int tmp_offset = 0;
    for (int process = 0; process < world_size; ++process) {
      int count_rows_for_process = len_counts[process];
      int start_row = len_displs[process];

      int tmp_count = 0;
      for (int index = 0; index < count_rows_for_process; ++index) {
        tmp_count += static_cast<int>(matrix[start_row + index].size());
      }
      val_counts[process] = tmp_count;
      val_displs[process] = tmp_offset;
      tmp_offset += tmp_count;
    }
  }

  MPI_Bcast(val_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(val_displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> flat_matrix;
  if (world_rank == 0) {
    int total_elements = val_displs[world_size - 1] + val_counts[world_size - 1];
    flat_matrix.reserve(total_elements);

    for (const auto &row : matrix) {
      flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
  }

  int my_elem_count = val_counts[world_rank];
  local_values.resize(my_elem_count);

  MPI_Scatterv(flat_matrix.data(), val_counts.data(), val_displs.data(), MPI_INT, local_values.data(), my_elem_count,
               MPI_INT, 0, MPI_COMM_WORLD);
}

void KiselevITestTaskMPI::ComputeLocalMax(const std::vector<int> &local_values,
                                          const std::vector<int> &local_row_lengths, std::vector<int> &local_result) {
  local_result.resize(local_row_lengths.size());

  if (row.empty()) {
    return;
  }
  out.resize(row.size());
  for (int row = 0, pos = 0; row < static_cast<int>(local_row_lengths.size()); ++row) {
    int len = local_row_lengths[row];
    int tmp_max = local_values[pos];
    for (int j = 1; j < len; ++j) {
      tmp_max = std::max(local_values[pos + j], tmp_max);
    }
    local_result[row] = tmp_max;
    pos += len;
  }
}

bool KiselevITestTaskMPI::RunImpl() {
  const auto &matrix = GetInput();
  auto &result_vector = GetOutput();

  int world_rank{0};
  int world_size{0};
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // 1. Передаём количество строк
  int count_all_rows = static_cast<int>(matrix.size());
  MPI_Bcast(&count_all_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // 2. Считаем локальное количество строк
  int local_row_count = CalculateLocalRowCount(count_all_rows, world_rank, world_size);

  // 3. Раздаём длины строк
  std::vector<int> local_row_lengths(local_row_count);
  std::vector<int> len_counts(world_size);
  std::vector<int> len_displs(world_size);
  DistributeRowLengths(matrix, count_all_rows, world_rank, world_size, local_row_lengths, len_counts, len_displs);

  // 4. Раздаём элементы
  std::vector<int> local_values;
  DistributeValues(matrix, world_rank, world_size, len_counts, len_displs, local_values);

  // 5. Поиск локальных максимумов
  std::vector<int> local_result;
  ComputeLocalMax(local_values, local_row_lengths, local_result);

  // 6. Собираем данные
  if (world_rank == 0) {
    result_vector.resize(count_all_rows);
  }
  MPI_Gatherv(local_result.data(), local_row_count, MPI_INT, result_vector.data(), len_counts.data(), len_displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(result_vector.data(), count_all_rows, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_max_value_in_strings
