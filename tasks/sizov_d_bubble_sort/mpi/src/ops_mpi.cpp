#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"

namespace sizov_d_bubble_sort {

namespace {

void ComputeScatterInfo(int total, int size, std::vector<int> &counts, std::vector<int> &displs) {
  const int base = total / size;
  const int rem = total % size;

  int offset = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }
}

void OddEvenPhase(std::vector<int> &local, const std::vector<int> &counts, const std::vector<int> &displs, int rank,
                  int size, int phase) {
  const int local_n = static_cast<int>(local.size());
  if (local_n == 0) {
    return;
  }

  const int parity = phase % 2;
  const int global_start = displs[rank];

  for (int i = 0; i + 1 < local_n; ++i) {
    const int g_idx = global_start + i;
    if ((g_idx % 2) == parity && local[i] > local[i + 1]) {
      std::swap(local[i], local[i + 1]);
    }
  }

  for (int step = 0; step < 2; ++step) {
    int partner = MPI_PROC_NULL;
    bool is_left_side = false;

    if (rank > 0 && counts[rank] > 0 && counts[rank - 1] > 0 && partner == MPI_PROC_NULL) {
      const int left_rank = rank - 1;
      const int boundary_index = displs[rank] - 1;

      if ((boundary_index % 2) == parity && (left_rank % 2) == step) {
        partner = left_rank;
        is_left_side = false;
      }
    }

    if (rank < size - 1 && counts[rank] > 0 && counts[rank + 1] > 0 && partner == MPI_PROC_NULL) {
      const int left_rank = rank;
      const int boundary_index = displs[rank + 1] - 1;

      if ((boundary_index % 2) == parity && (left_rank % 2) == step) {
        partner = rank + 1;
        is_left_side = true;
      }
    }

    if (partner == MPI_PROC_NULL) {
      continue;
    }

    int send_val = is_left_side ? local[local_n - 1] : local[0];
    int recv_val = 0;

    MPI_Sendrecv(&send_val, 1, MPI_INT, partner, 0, &recv_val, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    if (is_left_side) {
      local[local_n - 1] = std::min(send_val, recv_val);
    } else {
      local[0] = std::max(send_val, recv_val);
    }
  }
}

}  // namespace

// ============================================================================

SizovDBubbleSortMPI::SizovDBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool SizovDBubbleSortMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = (rank == 0 ? static_cast<int>(data_.size()) : 0);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank != 0) {
      data_.assign(n, 0);
    }
    if (n > 0) {
      MPI_Bcast(data_.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = data_;
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  ComputeScatterInfo(n, size, counts, displs);

  int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv((rank == 0 ? data_.data() : nullptr), counts.data(), displs.data(), MPI_INT, local.data(), local_n,
               MPI_INT, 0, MPI_COMM_WORLD);

  for (int phase = 0; phase < n; ++phase) {
    OddEvenPhase(local, counts, displs, rank, size, phase);
  }

  std::vector<int> result;
  if (rank == 0) {
    result.resize(n);
  }

  MPI_Gatherv(local.data(), local_n, MPI_INT, (rank == 0 ? result.data() : nullptr), counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  int out_n = (rank == 0 ? n : 0);
  MPI_Bcast(&out_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = result;
  } else {
    GetOutput().assign(out_n, 0);
  }

  if (out_n > 0) {
    MPI_Bcast(GetOutput().data(), out_n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool SizovDBubbleSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_bubble_sort
