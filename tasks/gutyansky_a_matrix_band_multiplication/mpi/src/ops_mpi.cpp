#include "gutyansky_a_matrix_band_multiplication/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gutyansky_a_matrix_band_multiplication/common/include/common.hpp"

namespace gutyansky_a_matrix_band_multiplication {

namespace {}

GutyanskyAMatrixBandMultiplicationMPI::GutyanskyAMatrixBandMultiplicationMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  GetInput() = in;
  GetOutput() = {};
}

void GutyanskyAMatrixBandMultiplicationMPI::GetScatterParams(int rank, int world_size, int elements_count, int *size,
                                                             int *displacement) {
  int chunk_size = elements_count / world_size;
  int remainder_size = elements_count % world_size;

  if (rank < remainder_size) {
    *size = chunk_size + 1;
    *displacement = rank * (chunk_size + 1);
  } else {
    *size = chunk_size;
    *displacement = (remainder_size * (chunk_size + 1)) + ((rank - remainder_size) * chunk_size);
  }
}

bool GutyanskyAMatrixBandMultiplicationMPI::ValidationImpl() {
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (!GetInput().first.IsValid()) {
      return false;
    }
    if (!GetInput().second.IsValid()) {
      return false;
    }

    return GetInput().first.cols == GetInput().second.rows;
  }

  return true;
}

bool GutyanskyAMatrixBandMultiplicationMPI::PreProcessingImpl() {
  return true;
}

std::vector<int> GutyanskyAMatrixBandMultiplicationMPI::ScatterA(int rank, int world_size, int rows_a, int cols_a) {
  int local_rows_a;
  int local_start_a;
  GetScatterParams(rank, world_size, rows_a, &local_rows_a, &local_start_a);

  std::vector<int> local_chunk_a(local_rows_a * cols_a);

  if (rank == 0) {
    std::vector<int> sizes_a(world_size);
    std::vector<int> displs_a(world_size);
    for (int i = 0; i < world_size; i++) {
      GetScatterParams(i, world_size, rows_a, &sizes_a[i], &displs_a[i]);
      sizes_a[i] *= cols_a;
      displs_a[i] *= cols_a;
    }

    MPI_Scatterv(GetInput().first.data.data(), sizes_a.data(), displs_a.data(), MPI_INT, local_chunk_a.data(),
                 static_cast<int>(local_chunk_a.size()), MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_chunk_a.data(), static_cast<int>(local_chunk_a.size()),
                 MPI_INT, 0, MPI_COMM_WORLD);
  }

  return local_chunk_a;
}

std::vector<int> GutyanskyAMatrixBandMultiplicationMPI::ScatterB(int rank, int world_size, int rows_b, int cols_b) {
  int local_cols_b;
  int local_start_b;
  GetScatterParams(rank, world_size, cols_b, &local_cols_b, &local_start_b);

  std::vector<int> local_chunk_b(rows_b * local_cols_b);

  if (rank == 0) {
    std::vector<int> sizes_b(world_size);
    std::vector<int> displs_b(world_size);
    for (int i = 0; i < world_size; i++) {
      GetScatterParams(i, world_size, cols_b, &sizes_b[i], &displs_b[i]);
      sizes_b[i] *= rows_b;
      displs_b[i] *= rows_b;
    }

    std::vector<int> packed_b;
    packed_b.reserve(GetInput().second.data.size());
    for (int i = 0; i < world_size; i++) {
      int start_col;
      int col_cnt;
      GetScatterParams(i, world_size, cols_b, &col_cnt, &start_col);

      for (int k = 0; k < GetInput().second.rows; k++) {
        for (int j = 0; j < col_cnt; j++) {
          packed_b.push_back(GetInput().second.data[(k * GetInput().second.cols) + start_col + j]);
        }
      }
    }

    MPI_Scatterv(packed_b.data(), sizes_b.data(), displs_b.data(), MPI_INT, local_chunk_b.data(),
                 static_cast<int>(local_chunk_b.size()), MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT, local_chunk_b.data(), static_cast<int>(local_chunk_b.size()),
                 MPI_INT, 0, MPI_COMM_WORLD);
  }

  return local_chunk_b;
}

void GutyanskyAMatrixBandMultiplicationMPI::GatherResult(int rank, int world_size, int rows_a, int cols_b,
                                                         std::vector<int> &res_buffer) {
  if (rank == 0) {
    GetOutput().rows = rows_a;
    GetOutput().cols = cols_b;
    GetOutput().data.resize(rows_a * cols_b);

    std::vector<int> sizes_res(world_size);
    std::vector<int> displs_res(world_size);
    for (int i = 0; i < world_size; i++) {
      GetScatterParams(i, world_size, rows_a, &sizes_res[i], &displs_res[i]);
      sizes_res[i] *= cols_b;
      displs_res[i] *= cols_b;
    }

    MPI_Gatherv(res_buffer.data(), static_cast<int>(res_buffer.size()), MPI_INT, GetOutput().data.data(),
                sizes_res.data(), displs_res.data(), MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(res_buffer.data(), static_cast<int>(res_buffer.size()), MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0,
                MPI_COMM_WORLD);
  }
}

bool GutyanskyAMatrixBandMultiplicationMPI::RunImpl() {
  int world_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t bc_sizes[3];

  if (rank == 0) {
    bc_sizes[0] = GetInput().first.rows;
    bc_sizes[1] = GetInput().first.cols;
    bc_sizes[2] = GetInput().second.cols;
  }

  MPI_Bcast(bc_sizes, 3, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  int local_rows_a;
  int local_start_a;
  GetScatterParams(rank, world_size, bc_sizes[0], &local_rows_a, &local_start_a);
  int local_cols_b;
  int local_start_b;
  GetScatterParams(rank, world_size, bc_sizes[2], &local_cols_b, &local_start_b);

  int rows_a = bc_sizes[0];
  int cols_a = bc_sizes[1];
  int rows_b = bc_sizes[1];
  int cols_b = bc_sizes[2];

  std::vector<int> local_chunk_a = ScatterA(rank, world_size, rows_a, cols_a);
  std::vector<int> local_chunk_b = ScatterB(rank, world_size, rows_b, cols_b);
  std::vector<int> res_buffer(local_rows_a * cols_b, 0);
  std::vector<int> rotation_buffer;

  for (int it = 0; it < world_size; it++) {
    int col_count;
    int start_col;
    GetScatterParams((rank + it) % world_size, world_size, cols_b, &col_count, &start_col);

    for (int i = 0; i < local_rows_a; i++) {
      for (int j = 0; j < col_count; j++) {
        for (int k = 0; k < cols_a; k++) {
          res_buffer[(i * cols_b) + start_col + j] +=
              local_chunk_a[(i * cols_a) + k] * local_chunk_b[(k * col_count) + j];
        }
      }
    }

    int next_col_size;
    int next_col_start;
    GetScatterParams((rank + it + 1) % world_size, world_size, cols_b, &next_col_size, &next_col_start);
    rotation_buffer.resize(rows_b * next_col_size);
    MPI_Sendrecv(local_chunk_b.data(), static_cast<int>(local_chunk_b.size()), MPI_INT,
                 (world_size + rank - 1) % world_size, 0, rotation_buffer.data(),
                 static_cast<int>(rotation_buffer.size()), MPI_INT, (rank + 1) % world_size, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    std::swap(local_chunk_b, rotation_buffer);
  }

  GatherResult(rank, world_size, rows_a, cols_b, res_buffer);

  return true;
}

bool GutyanskyAMatrixBandMultiplicationMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gutyansky_a_matrix_band_multiplication
