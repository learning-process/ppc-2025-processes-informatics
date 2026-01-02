#include "volkov_a_radix_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "volkov_a_radix_batcher/common/include/common.hpp"

namespace volkov_a_radix_batcher {

VolkovARadixBatcherMPI::VolkovARadixBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VolkovARadixBatcherMPI::ValidationImpl() {
  return true;
}

bool VolkovARadixBatcherMPI::PreProcessingImpl() {
  return true;
}

uint64_t VolkovARadixBatcherMPI::DoubleToOrderedInt(double d) {
  uint64_t u = 0;
  std::memcpy(&u, &d, sizeof(d));
  uint64_t mask = (static_cast<uint64_t>(1) << 63);
  if ((u & mask) != 0) {
    return ~u;
  }
  return u | mask;
}

double VolkovARadixBatcherMPI::OrderedIntToDouble(uint64_t k) {
  uint64_t mask = (static_cast<uint64_t>(1) << 63);
  if ((k & mask) != 0) {
    k &= ~mask;
  } else {
    k = ~k;
  }
  double d = 0.0;
  std::memcpy(&d, &k, sizeof(d));
  return d;
}

void VolkovARadixBatcherMPI::RadixSortDouble(std::vector<double> &data) {
  if (data.empty()) {
    return;
  }

  std::vector<uint64_t> keys(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    keys[i] = DoubleToOrderedInt(data[i]);
  }

  std::vector<uint64_t> temp(data.size());
  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<size_t> counts(256, 0);
    for (uint64_t k : keys) {
      counts[(k >> shift) & 0xFF]++;
    }

    std::vector<size_t> positions(256);
    positions[0] = 0;
    for (int i = 1; i < 256; i++) {
      positions[i] = positions[i - 1] + counts[i - 1];
    }

    for (uint64_t k : keys) {
      temp[positions[(k >> shift) & 0xFF]++] = k;
    }
    keys = temp;
  }

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = OrderedIntToDouble(keys[i]);
  }
}

void VolkovARadixBatcherMPI::CalculateDistribution(int world_size, int total_elements, std::vector<int> &counts,
                                                   std::vector<int> &displs) {
  counts.resize(world_size);
  displs.resize(world_size);

  int base_size = total_elements / world_size;
  int remainder = total_elements % world_size;

  int current_displ = 0;
  for (int i = 0; i < world_size; ++i) {
    counts[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = current_displ;
    current_displ += counts[i];
  }
}

bool VolkovARadixBatcherMPI::RunImpl() {
  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (world_size == 1) {
    auto data = GetInput();
    RadixSortDouble(data);
    GetOutput() = data;
    return true;
  }

  int total_elements = 0;
  if (rank == 0) {
    total_elements = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_elements == 0) {
    return true;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  CalculateDistribution(world_size, total_elements, counts, displs);

  std::vector<double> local_vec(counts[rank]);

  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), counts.data(), displs.data(), MPI_DOUBLE, local_vec.data(), counts[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, counts.data(), displs.data(), MPI_DOUBLE, local_vec.data(), counts[rank], MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }

  RadixSortDouble(local_vec);
  ParallelMergeSort(rank, world_size, counts, local_vec);

  if (rank == 0) {
    std::vector<double> result(total_elements);
    MPI_Gatherv(local_vec.data(), counts[rank], MPI_DOUBLE, result.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    GetOutput() = result;
  } else {
    MPI_Gatherv(local_vec.data(), counts[rank], MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  return true;
}

void VolkovARadixBatcherMPI::ParallelMergeSort(int rank, int world_size, const std::vector<int> &counts,
                                               std::vector<double> &local_vec) {
  int max_count = 0;
  for (int c : counts) {
    max_count = std::max(max_count, c);
  }

  std::vector<double> buffer_recv(max_count);
  std::vector<double> buffer_merge(local_vec.size() + max_count);

  for (int phase = 0; phase < world_size; ++phase) {
    int neighbor = -1;

    if ((rank % 2) == (phase % 2)) {
      neighbor = rank + 1;
    } else {
      neighbor = rank - 1;
    }

    if (neighbor >= 0 && neighbor < world_size) {
      ExchangeAndMerge(rank, neighbor, counts, local_vec, buffer_recv, buffer_merge);
    }
  }
}

void VolkovARadixBatcherMPI::ExchangeAndMerge(int rank, int neighbor, const std::vector<int> &counts,
                                              std::vector<double> &local_vec, std::vector<double> &buffer_recv,
                                              std::vector<double> &buffer_merge) {
  MPI_Sendrecv(local_vec.data(), counts[rank], MPI_DOUBLE, neighbor, 0, buffer_recv.data(), counts[neighbor],
               MPI_DOUBLE, neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::merge(local_vec.begin(), local_vec.end(), buffer_recv.begin(), buffer_recv.begin() + counts[neighbor],
             buffer_merge.begin());

  if (rank < neighbor) {
    std::copy(buffer_merge.begin(), buffer_merge.begin() + counts[rank], local_vec.begin());
  } else {
    int my_start_idx = counts[neighbor];
    std::copy(buffer_merge.begin() + my_start_idx, buffer_merge.begin() + my_start_idx + counts[rank],
              local_vec.begin());
  }
}

bool VolkovARadixBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_radix_batcher
