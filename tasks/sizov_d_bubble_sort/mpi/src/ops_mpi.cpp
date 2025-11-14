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

bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total = static_cast<int>(data_.size());
  const int base = total / size;
  const int rem = total % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  int offset = 0;
  int idx = 0;
  for (auto &c : counts) {
    c = base;
    if (idx < rem) {
      ++c;
    }
    displs[idx] = offset;
    offset += c;
    ++idx;
  }

  const int local_n = counts[rank];
  std::vector<int> local(local_n);

  MPI_Scatterv(data_.data(), counts.data(), displs.data(), MPI_INT, local.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

  std::ranges::sort(local);

  for (int phase = 0; phase < size; ++phase) {
    const bool even_phase = ((phase % 2) == 0);
    const bool even_rank = ((rank % 2) == 0);

    int partner = -1;
    if (even_phase) {
      if (even_rank) {
        partner = rank + 1;
      } else {
        partner = rank - 1;
      }
    } else {
      if (!even_rank) {
        partner = rank + 1;
      } else {
        partner = rank - 1;
      }
    }

    if (partner < 0 || partner >= size) {
      continue;
    }

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

  std::vector<int> global;
  if (rank == 0) {
    global.resize(total);
  }

  MPI_Gatherv(local.data(), local_n, MPI_INT, (rank == 0 ? global.data() : nullptr),
              (rank == 0 ? counts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::ranges::sort(global);
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
