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

// --- Вспомогательные функции ---
int KiselevITestTaskMPI::CalculateLocalRowCount(int total_rows, int world_rank, int world_size) {
  int base = total_rows / world_size;
  int rem = total_rows % world_size;
  return base + (world_rank < rem ? 1 : 0);
}

void KiselevITestTaskMPI::DistributeRowLengths(const std::vector<std::vector<int>> &matrix, int total_rows,
                                               int world_rank, int world_size, std::vector<int> &local_row_lengths,
                                               std::vector<int> &len_counts, std::vector<int> &len_displs) {
  std::vector<int> all_row_lengths;
  int base = total_rows / world_size;
  int rem = total_rows % world_size;

  if (world_rank == 0) {
    all_row_lengths.resize(static_cast<size_t>(total_rows));
    for (int i = 0; i < total_rows; ++i) {
      all_row_lengths[static_cast<size_t>(i)] = static_cast<int>(matrix[static_cast<size_t>(i)].size());
    }

    int offset = 0;
    for (int p = 0; p < world_size; ++p) {
      len_counts[p] = base + (p < rem ? 1 : 0);
      len_displs[p] = offset;
      offset += len_counts[p];
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
    int offset = 0;
    for (int p = 0; p < world_size; ++p) {
      int count = 0;
      for (int i = 0; i < len_counts[p]; ++i) {
        count += static_cast<int>(matrix[len_displs[p] + i].size());
      }
      val_counts[p] = count;
      val_displs[p] = offset;
      offset += count;
    }
  }

  MPI_Bcast(val_counts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(val_displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> flat_matrix;
  if (world_rank == 0) {
    int total_elements = val_displs[world_size - 1] + val_counts[world_size - 1];
    flat_matrix.reserve(static_cast<size_t>(total_elements));
    for (const auto &row : matrix) {
      flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
  }

  int my_count = val_counts[world_rank];
  if (my_count > 0) {
    local_values.resize(my_count);
  } else {
    local_values.clear();
  }

  MPI_Scatterv(flat_matrix.data(), val_counts.data(), val_displs.data(), MPI_INT, local_values.data(), my_count,
               MPI_INT, 0, MPI_COMM_WORLD);
}

void KiselevITestTaskMPI::ComputeLocalMax(const std::vector<int> &local_values,
                                          const std::vector<int> &local_row_lengths, std::vector<int> &local_result) {
  size_t n_rows = local_row_lengths.size();
  local_result.resize(n_rows);
  if (n_rows == 0) {
    return;
  }

  int pos = 0;
  for (size_t r = 0; r < n_rows; ++r) {
    int len = local_row_lengths[r];
    if (len == 0) {
      local_result[r] = std::numeric_limits<int>::min();
      continue;
    }
    int tmp_max = local_values[pos];
    for (int j = 1; j < len; ++j) {
      tmp_max = std::max(tmp_max, local_values[pos + j]);
    }
    local_result[r] = tmp_max;
    pos += len;
  }
}

bool KiselevITestTaskMPI::RunImpl() {
  const auto &matrix = GetInput();
  auto &result_vector = GetOutput();

  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_rows = static_cast<int>(matrix.size());
  MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_row_count = CalculateLocalRowCount(total_rows, world_rank, world_size);

  std::vector<int> local_row_lengths(local_row_count);
  std::vector<int> len_counts(world_size);
  std::vector<int> len_displs(world_size);
  DistributeRowLengths(matrix, total_rows, world_rank, world_size, local_row_lengths, len_counts, len_displs);

  std::vector<int> local_values;
  DistributeValues(matrix, world_rank, world_size, len_counts, len_displs, local_values);

  std::vector<int> local_result;
  ComputeLocalMax(local_values, local_row_lengths, local_result);

  if (world_rank == 0) {
    result_vector.resize(total_rows);
  }

  MPI_Gatherv(local_result.data(), local_row_count, MPI_INT, result_vector.data(), len_counts.data(), len_displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(result_vector.data(), total_rows, MPI_INT, 0, MPI_COMM_WORLD);
  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_max_value_in_strings
