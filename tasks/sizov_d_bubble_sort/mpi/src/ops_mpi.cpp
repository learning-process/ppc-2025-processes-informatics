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
  for (int i = 0; i < size; i++) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }
}

int ComputePartner(bool even_phase, bool even_rank, int rank) {
  return (even_phase == even_rank) ? rank + 1 : rank - 1;
}

void OddEvenExchange(std::vector<int> &local, const std::vector<int> &counts, int rank, int size, int phase) {
  const bool even_phase = (phase % 2 == 0);
  const bool even_rank = (rank % 2 == 0);

  const int partner = ComputePartner(even_phase, even_rank, rank);
  if (partner < 0 || partner >= size) {
    return;
  }

  const int local_n = static_cast<int>(local.size());
  const int partner_n = counts[partner];

  if (local_n == 0 || partner_n == 0) {
    return;
  }

  std::vector<int> recvbuf(partner_n);

  MPI_Sendrecv(local.data(), local_n, MPI_INT, partner, 0, recvbuf.data(), partner_n, MPI_INT, partner, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> merged(local_n + partner_n);
  std::ranges::merge(local, recvbuf, merged.begin());

  if (rank < partner) {
    std::copy(merged.begin(), merged.begin() + local_n, local.begin());
  } else {
    std::copy(merged.end() - local_n, merged.end(), local.begin());
  }
}

}  // namespace

SizovDBubbleSortMPI::SizovDBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    return !input.empty();
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

  const int total = (rank == 0 ? static_cast<int>(data_.size()) : 0);

  int n = total;
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

  std::vector<int> counts(size), displs(size);
  ComputeScatterInfo(n, size, counts, displs);

  const int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv((rank == 0 ? data_.data() : nullptr), counts.data(), displs.data(), MPI_INT, local.data(), local_n,
               MPI_INT, 0, MPI_COMM_WORLD);

  if (local_n > 1) {
    std::ranges::sort(local);
  }

  for (int phase = 0; phase < n; ++phase) {
    OddEvenExchange(local, counts, rank, size, phase);
  }

  std::vector<int> result;
  if (rank == 0) {
    result.resize(n);
  }

  MPI_Gatherv(local.data(), local_n, MPI_INT, (rank == 0 ? result.data() : nullptr), counts.data(), displs.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  int out_n = 0;
  if (rank == 0) {
    out_n = n;
  }
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
