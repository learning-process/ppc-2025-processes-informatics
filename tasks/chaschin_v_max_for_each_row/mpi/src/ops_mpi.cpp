#include "chaschin_v_max_for_each_row/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"
#include "util/include/util.hpp"

namespace chaschin_v_max_for_each_row {

ChaschinVMaxForEachRow::ChaschinVMaxForEachRow(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
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
  // int end = start + count;

  // ----------------------------
  // Exchange rows with explicit per-row handshake (length -> data)
  // ----------------------------
  std::vector<std::vector<float>> local_mat(count);

// DEBUG: print assignment info
#ifndef NDEBUG
  {
    int dbg_rank = rank;
    fprintf(stderr, "DEBUG rank=%d nrows=%d size=%d base=%d rem=%d start=%d count=%d\n", dbg_rank, nrows, size, base,
            rem, start, count);
    fflush(stderr);
  }
#endif

  if (rank == 0) {
    // root: for each other rank, send rows assigned to that rank
    for (int p = 1; p < size; ++p) {
      int p_start = p * base + std::min(p, rem);
      int p_count = base + (p < rem ? 1 : 0);
      if (p_count <= 0) {
        continue;
      }

      for (int i = 0; i < p_count; ++i) {
        const auto &row = mat[p_start + i];
        int len = static_cast<int>(row.size());
        // send length first (tag 100)
        MPI_Send(&len, 1, MPI_INT, p, 100, MPI_COMM_WORLD);
        // then send data if any (tag 101)
        if (len > 0) {
          MPI_Send(row.data(), len, MPI_FLOAT, p, 101, MPI_COMM_WORLD);
        }
      }
    }

    // copy own rows into local_mat
    for (int i = 0; i < count; ++i) {
      if (!mat[start + i].empty()) {
        local_mat[i] = mat[start + i];
      } else {
        local_mat[i].clear();
      }
    }

  } else {
    // worker: receive p_count rows (length then data for each)
    if (count > 0) {
      for (int i = 0; i < count; ++i) {
        int len = 0;
        MPI_Recv(&len, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (len > 0) {
          local_mat[i].resize(len);
          MPI_Recv(local_mat[i].data(), len, MPI_FLOAT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
          local_mat[i].clear();
        }
      }
    }
  }

// optional barrier to make debugging deterministic
#ifndef NDEBUG
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // ----------------------------
  // Compute local maxima
  // ----------------------------

  std::vector<float> local_out(count);

  for (int i = 0; i < count; i++) {
    if (!local_mat[i].empty()) {
      local_out[i] = *std::max_element(local_mat[i].begin(), local_mat[i].end());
    } else {
      local_out[i] = std::numeric_limits<float>::lowest();  // или 0, по логике задачи
    }
  }

  // ----------------------------
  // Send local results to root
  // ----------------------------
  if (rank == 0) {
    auto &out = GetOutput();
    out.resize(nrows);
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
    if (count > 0) {
      MPI_Send(local_out.data(), count, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
  }

  // --- BROADCAST final output from root to all ranks so CheckTestOutputData passes on every rank ---
  // Ensure out exists on root and allocate on workers
  if (rank == 0) {
    // out is already filled on root and resized to nrows earlier
    // nothing to do here
  } else {
    // Workers must have a vector with correct size and contiguous storage
    GetOutput().assign(nrows, 0.0f);
  }

  if (nrows > 0) {
    // Broadcast raw float data (contiguous)
    MPI_Bcast(GetOutput().data(), nrows, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool ChaschinVMaxForEachRow::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  return !GetOutput().empty();
}
}  // namespace chaschin_v_max_for_each_row
