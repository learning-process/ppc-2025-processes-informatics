#include "sizov_d_bubble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <ranges>
#include <vector>

#include "sizov_d_bubble_sort/common/include/common.hpp"

namespace sizov_d_bubble_sort {

SizovDBubbleSortMPI::SizovDBubbleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SizovDBubbleSortMPI::ValidationImpl() {
  const auto &input = GetInput();
  return !input.empty();
}

bool SizovDBubbleSortMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

static void ComputeScatterInfo(int total, int size, int rem, std::vector<int> &counts, std::vector<int> &displs) {
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    const int chunk = total / size + (i < rem ? 1 : 0);
    counts[i] = chunk;
    displs[i] = offset;
    offset += chunk;
  }
}

static void OddEvenExchange(std::vector<int> &local, const std::vector<int> &counts, int rank, int size, int phase) {
  const bool even_phase = (phase % 2 == 0);
  const bool even_rank = (rank % 2 == 0);

  int partner = even_phase ? (even_rank ? rank + 1 : rank - 1) : (!even_rank ? rank + 1 : rank - 1);

  if (partner < 0 || partner >= size) {
    return;
  }

  const int local_n = static_cast<int>(local.size());
  const int pn = counts[partner];

  std::vector<int> recvbuf(pn);

  MPI_Sendrecv(local.data(), local_n, MPI_INT, partner, 0, recvbuf.data(), pn, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<int> merged(local_n + pn);
  std::ranges::merge(local, recvbuf, merged.begin());

  if (rank < partner) {
    std::ranges::copy(merged | std::views::take(local_n), local.begin());
  } else {
    std::ranges::copy(merged | std::views::drop(pn), local.begin());
  }
}

static void GatherResult(const std::vector<int> &local, const std::vector<int> &counts, const std::vector<int> &displs,
                         int rank, int total, std::vector<int> &output) {
  if (rank == 0) {
    output.resize(total);
  }

  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_INT, (rank == 0 ? output.data() : nullptr),
              (rank == 0 ? counts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::ranges::sort(output);
  }
}

bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total = static_cast<int>(data_.size());
  const int rem = total % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  ComputeScatterInfo(total, size, rem, counts, displs);

  const int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv(data_.data(), counts.data(), displs.data(), MPI_INT, local.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

  std::ranges::sort(local);

  for (int phase = 0; phase < size; ++phase) {
    OddEvenExchange(local, counts, rank, size, phase);
  }

  std::vector<int> global;
  GatherResult(local, counts, displs, rank, total, global);

  if (rank == 0) {
    GetOutput() = global;
  }

  return true;
}

bool SizovDBubbleSortMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n = (rank == 0) ? static_cast<int>(GetOutput().size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().assign(n, 0);
  }

  if (n > 0) {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

}  // namespace sizov_d_bubble_sort
