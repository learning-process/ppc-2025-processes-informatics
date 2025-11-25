#include "kondakov_v_min_val_in_matrix_str/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "kondakov_v_min_val_in_matrix_str/common/include/common.hpp"

namespace kondakov_v_min_val_in_matrix_str {

KondakovVMinValMatrixMPI::KondakovVMinValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
  GetOutput().resize(in.size());
}

bool KondakovVMinValMatrixMPI::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return true;
  }

  size_t cols = matrix[0].size();
  // NOLINTNEXTLINE(readability-use-anyofallof)
  for (const auto &row : matrix) {
    if (row.empty()) {
      return false;
    }
    if (row.size() != cols) {
      return false;
    }
  }
  return true;
}

bool KondakovVMinValMatrixMPI::PreProcessingImpl() {
  return true;
}

bool KondakovVMinValMatrixMPI::RunImpl() {
  int n = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &in_data = GetInput();

  size_t total_rows = 0;
  size_t cols = 0;
  BroadcastMatrixMetadata(in_data, total_rows, cols);

  if (total_rows == 0) {
    HandleEmptyMatrix(rank);
    return true;
  }

  int st_row = 0;
  int en_row = 0;
  ComputeRowRanges(n, rank, total_rows, st_row, en_row);

  std::vector<int> temp;
  FindLocalMinima(in_data, st_row, en_row, cols, temp);

  int extra = static_cast<int>(total_rows % n);
  std::vector<int> recv_counts(n, 0);
  std::vector<int> displs(n, 0);
  PrepareGathervParams(n, total_rows, extra, recv_counts, displs);

  if (rank == 0) {
    GetOutput().resize(total_rows);
  }

  int *recv_buffer = (rank == 0) ? GetOutput().data() : nullptr;
  MPI_Gatherv(temp.data(), static_cast<int>(temp.size()), MPI_INT, recv_buffer, recv_counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(GetOutput().data(), static_cast<int>(total_rows), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void KondakovVMinValMatrixMPI::BroadcastMatrixMetadata(const InType &in_data, size_t &total_rows, size_t &cols) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  uint64_t total_rows_u64 = (rank == 0) ? static_cast<uint64_t>(in_data.size()) : 0;
  uint64_t cols_u64 = (rank == 0 && total_rows_u64 > 0) ? static_cast<uint64_t>(in_data[0].size()) : 0;

  MPI_Bcast(&total_rows_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_u64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  total_rows = static_cast<size_t>(total_rows_u64);
  cols = static_cast<size_t>(cols_u64);
}

void KondakovVMinValMatrixMPI::HandleEmptyMatrix(int rank) {
  if (rank == 0) {
    GetOutput().clear();
  }
  std::vector<int> empty;
  MPI_Bcast(empty.data(), 0, MPI_INT, 0, MPI_COMM_WORLD);
}

void KondakovVMinValMatrixMPI::ComputeRowRanges(int n, int rank, size_t total_rows, int &st_row, int &en_row) {
  int delta = static_cast<int>(total_rows / n);
  int extra = static_cast<int>(total_rows % n);
  st_row = (rank * delta) + std::min(rank, extra);
  en_row = ((rank + 1) * delta) + std::min(rank + 1, extra);
}

void KondakovVMinValMatrixMPI::FindLocalMinima(const InType &in_data, int st_row, int en_row, size_t cols,
                                               std::vector<int> &temp) {
  temp.resize(en_row - st_row);
  for (size_t i = 0; i < temp.size(); ++i) {
    size_t global_row = st_row + i;
    int min_val = std::numeric_limits<int>::max();
    for (size_t j = 0; j < cols; ++j) {
      min_val = std::min(min_val, in_data[global_row][j]);
    }
    temp[i] = min_val;
  }
}

void KondakovVMinValMatrixMPI::PrepareGathervParams(int n, size_t total_rows, int extra, std::vector<int> &recv_counts,
                                                    std::vector<int> &displs) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    for (int i = 0; i < n; ++i) {
      int add = ((i < extra) ? 1 : 0);
      recv_counts[i] = static_cast<int>((total_rows / n) + add);
    }
    displs[0] = 0;
    for (int i = 1; i < n; ++i) {
      displs[i] = displs[i - 1] + recv_counts[i - 1];
    }
  }

  MPI_Bcast(recv_counts.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
}

bool KondakovVMinValMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kondakov_v_min_val_in_matrix_str
