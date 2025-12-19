#include "rozenberg_a_bubble_odd_even_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "rozenberg_a_bubble_odd_even_sort/common/include/common.hpp"

namespace rozenberg_a_bubble_odd_even_sort {

RozenbergABubbleOddEvenSortMPI::RozenbergABubbleOddEvenSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType empty;
  GetInput().swap(empty);

  for (const auto &elem : in) {
    GetInput().push_back(elem);
  }

  GetOutput().clear();
}

bool RozenbergABubbleOddEvenSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return (!(GetInput().empty())) && (GetOutput().empty());
  }
  return true;
}

bool RozenbergABubbleOddEvenSortMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().resize(GetInput().size());
  }
  return true;
}

bool RozenbergABubbleOddEvenSortMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  int sum = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[i] = n / size + (i < (n % size) ? 1 : 0);
    displs[i] = sum;
    sum += sendcounts[i];
  }

  int chunk = sendcounts[rank];
  InType local_buf(static_cast<size_t>(chunk));
  MPI_Scatterv(GetInput().data(), sendcounts.data(), displs.data(), MPI_INT, local_buf.data(), chunk, MPI_INT, 0,
               MPI_COMM_WORLD);

  for (int i = 0; i < chunk; i++) {
    for (int j = 0; j < chunk - 1; j++) {
      if (local_buf[j] > local_buf[j + 1]) {
        std::swap(local_buf[j], local_buf[j + 1]);
      }
    }
  }

  for (int i = 0; i <= size; i++) {
    int neighbor = -1;
    if (i % 2 == 0) {
      if (rank % 2 == 0) {
        neighbor = rank + 1;
      } else {
        neighbor = rank - 1;
      }
    } else {
      if (rank % 2 == 0) {
        neighbor = rank - 1;
      } else {
        neighbor = rank + 1;
      }
    }

    if (neighbor >= 0 && neighbor < size) {
      int neighbor_n = sendcounts[neighbor];
      std::vector<int> neighbor_data(static_cast<size_t>(neighbor_n));
      std::vector<int> merged(static_cast<size_t>(chunk + neighbor_n));

      MPI_Sendrecv(local_buf.data(), chunk, MPI_INT, neighbor, 0, neighbor_data.data(), neighbor_n, MPI_INT, neighbor,
                   0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::merge(local_buf.begin(), local_buf.end(), neighbor_data.begin(), neighbor_data.end(), merged.begin());

      if (rank < neighbor) {
        std::copy(merged.begin(), merged.begin() + chunk, local_buf.begin());
      } else {
        std::copy(merged.end() - chunk, merged.end(), local_buf.begin());
      }
    }
  }

  MPI_Gatherv(local_buf.data(), chunk, MPI_INT, GetOutput().data(), sendcounts.data(), displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  return true;
}

bool RozenbergABubbleOddEvenSortMPI::PostProcessingImpl() {
  return true;
}

}  // namespace rozenberg_a_bubble_odd_even_sort
