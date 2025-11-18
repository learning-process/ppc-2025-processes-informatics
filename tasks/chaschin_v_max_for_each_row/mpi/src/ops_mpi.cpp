#include "chaschin_v_max_for_each_row/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>
#include <algorithm>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"
#include "util/include/util.hpp"

namespace chaschin_v_max_for_each_row {

ChaschinVMaxForEachRow::ChaschinVMaxForEachRow(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  this->GetOutput().clear();
}

bool ChaschinVMaxForEachRow::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  const auto &mat = GetInput();
  if (mat.empty()) {
    return false;
  }

  for (const auto &row : mat) {
    if (row.empty()) {
      return false;
    }
  }
  return true;
}

bool ChaschinVMaxForEachRow::PreProcessingImpl() {
  return true;
}

bool ChaschinVMaxForEachRow::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &mat = GetInput();

  int nrows = 0;
  if (rank == 0) {
    nrows = mat.size();
  }

  // Broadcast number of rows
  MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Determine distribution of rows
  int base = nrows / size;
  int rem = nrows % size;

  int start = rank * base + std::min(rank, rem);
  int count = base + (rank < rem ? 1 : 0);
  //int end = start + count;

  // ----------------------------
  // Send/recv row sizes
  // ----------------------------
  std::vector<int> row_sizes(count);
  std::vector<std::vector<int>> all_row_sizes(size);

  if (rank == 0) {
    for (int p = 1; p < size; p++) {
        int p_start = p * base + std::min(p, rem);
        int p_count = base + (p < rem ? 1 : 0);
        all_row_sizes[p].resize(p_count);

        for (int i = 0; i < p_count; i++) {
            all_row_sizes[p][i] = mat[p_start + i].size();
        }

        MPI_Send(all_row_sizes[p].data(), p_count, MPI_INT, p, 0, MPI_COMM_WORLD);
    }

    // own sizes
    all_row_sizes[0].resize(count);
    for (int i = 0; i < count; i++) {
        all_row_sizes[0][i] = mat[start + i].size();
    }
  }
  else {
    if (count > 0) {
      MPI_Recv(row_sizes.data(), count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // ----------------------------
  // Receive actual rows
  // ----------------------------
  std::vector<std::vector<float>> local_mat(count);

  if (rank == 0) {
    for (int p = 1; p < size; p++) {
        int p_start = p * base + std::min(p, rem);
        int p_count = base + (p < rem ? 1 : 0);
        for (int i = 0; i < p_count; i++) {
            MPI_Send(mat[p_start + i].data(), all_row_sizes[p][i], MPI_FLOAT, p, 1, MPI_COMM_WORLD);
        }
    }

    // copy own rows
    for (int i = 0; i < count; i++) {
        local_mat[i] = mat[start + i];
    }
  }
  else {
    for (int i = 0; i < count; i++) {
      local_mat[i].resize(row_sizes[i]);
      MPI_Recv(local_mat[i].data(), row_sizes[i], MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // ----------------------------
  // Compute local maxima
  // ----------------------------
  std::vector<float> local_out(count);

  for (int i = 0; i < count; i++) {
    local_out[i] = *std::max_element(local_mat[i].begin(), local_mat[i].end());
  }

  // ----------------------------
  // Send local results to root
  // ----------------------------
  if (rank == 0) {
    auto &out = GetOutput();
    for (int i = 0; i < count; i++) {
      out[start + i] = local_out[i];
    }

    for (int p = 1; p < size; p++) {
      int p_start = p * base + std::min(p, rem);
      int p_count = base + (p < rem ? 1 : 0);

      if (p_count > 0) {
        std::vector<float> tmp(p_count);
        MPI_Recv(tmp.data(), p_count, MPI_FLOAT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < p_count; i++) {
          out[p_start + i] = tmp[i];
        }
      }
    }

  } else {
    MPI_Send(local_out.data(), count, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
  }

  return true;
}

bool ChaschinVMaxForEachRow::PostProcessingImpl() {
  return !GetOutput().empty();
}
}  // namespace chaschin_v_max_for_each_row
