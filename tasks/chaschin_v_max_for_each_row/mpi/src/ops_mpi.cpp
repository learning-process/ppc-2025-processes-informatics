#include "chaschin_v_max_for_each_row/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <ranges>
#include <utility>
#include <vector>

#include "chaschin_v_max_for_each_row/common/include/common.hpp"

namespace chaschin_v_max_for_each_row {

ChaschinVMaxForEachRow::ChaschinVMaxForEachRow(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;
  GetInput() = std::move(in_copy);
  this->GetOutput().clear();
}

bool ChaschinVMaxForEachRow::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  const auto &mat = GetInput();
  if (mat.empty()) {
    return false;
  }

  return std::ranges::all_of(mat, [](const auto &row) { return !row.empty(); });
}

bool ChaschinVMaxForEachRow::PreProcessingImpl() {
  return true;
}

chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::RowRange
chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::ComputeRange(int nrows, int rank, int size) {
  int base = nrows / size;
  int rem = nrows % size;
  int start = rank * base + std::min(rank, rem);
  int count = base + (rank < rem ? 1 : 0);
  return {start, count};
}

std::vector<std::vector<float>> chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::DistributeRows(
    const std::vector<std::vector<float>> &mat, int rank, int size, const RowRange &range) {
  std::vector<std::vector<float>> local_mat(range.count);

  if (rank == 0) {
    // root: send rows to other ranks
    for (int p = 1; p < size; ++p) {
      RowRange r = ComputeRange(mat.size(), p, size);
      for (int i = 0; i < r.count; ++i) {
        int len = static_cast<int>(mat[r.start + i].size());
        MPI_Send(&len, 1, MPI_INT, p, 100, MPI_COMM_WORLD);
        if (len > 0) {
          MPI_Send(mat[r.start + i].data(), len, MPI_FLOAT, p, 101, MPI_COMM_WORLD);
        }
      }
    }
    // copy own rows
    for (int i = 0; i < range.count; ++i) {
      local_mat[i] = mat[range.start + i];
    }
  } else {
    // worker: receive rows
    for (int i = 0; i < range.count; ++i) {
      int len;
      MPI_Recv(&len, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_mat[i].resize(len);
      if (len > 0) {
        MPI_Recv(local_mat[i].data(), len, MPI_FLOAT, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }

  return local_mat;
}

std::vector<float> chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::ComputeLocalMax {
  std::vector<float> local_out(local_mat.size());
  for (size_t i = 0; i < local_mat.size(); ++i) {
    local_out[i] = local_mat[i].empty() ? std::numeric_limits<float>::lowest()
                                        : *std::max_element(local_mat[i].begin(), local_mat[i].end());
  }
  return local_out;
}

void chaschin_v_max_for_each_row::ChaschinVMaxForEachRow::GatherResults(std::vector<float> &out,
                                                                        const std::vector<float> &local_out, int rank,
                                                                        int size, const RowRange &range) {
  if (rank == 0) {
    for (int i = 0; i < range.count; ++i) {
      out[range.start + i] = local_out[i];
    }

    for (int p = 1; p < size; ++p) {
      RowRange r = ComputeRange(out.size(), p, size);
      if (r.count > 0) {
        std::vector<float> tmp(r.count);
        MPI_Recv(tmp.data(), r.count, MPI_FLOAT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < r.count; ++i) {
          out[r.start + i] = tmp[i];
        }
      }
    }
  } else {
    if (!local_out.empty()) {
      MPI_Send(local_out.data(), static_cast<int>(local_out.size()), MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }
  }
}

bool ChaschinVMaxForEachRow::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &mat = GetInput();
  int nrows = (rank == 0) ? mat.size() : 0;
  MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  RowRange range = ComputeRange(nrows, rank, size);
  auto local_mat = DistributeRows(mat, rank, size, range);
  auto local_out = ComputeLocalMax(local_mat);

  if (rank == 0) {
    GetOutput().resize(nrows);
  }
  GatherResults(GetOutput(), local_out, rank, size, range);
  // --- ensure output buffer exists on all ranks before broadcasting ---
  auto &out = GetOutput();
  if (rank != 0) {
    out.resize(nrows);  // <- ключевая строка: выделяем память на воркерах
  }

  if (nrows > 0) {
    MPI_Bcast(out.data(), nrows, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
