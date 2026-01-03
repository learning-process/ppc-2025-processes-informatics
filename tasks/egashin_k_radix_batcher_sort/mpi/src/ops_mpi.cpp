#include "egashin_k_radix_batcher_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "egashin_k_radix_batcher_sort/common/include/common.hpp"

namespace egashin_k_radix_batcher_sort {

EgashinKRadixBatcherSortMPI::EgashinKRadixBatcherSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool EgashinKRadixBatcherSortMPI::ValidationImpl() {
  return true;
}

bool EgashinKRadixBatcherSortMPI::PreProcessingImpl() {
  return true;
}

bool EgashinKRadixBatcherSortMPI::PostProcessingImpl() {
  return true;
}

uint64_t EgashinKRadixBatcherSortMPI::DoubleToSortable(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(double));
  if ((bits & (1ULL << 63)) != 0) {
    bits = ~bits;
  } else {
    bits ^= (1ULL << 63);
  }
  return bits;
}

double EgashinKRadixBatcherSortMPI::SortableToDouble(uint64_t bits) {
  if ((bits & (1ULL << 63)) != 0) {
    bits ^= (1ULL << 63);
  } else {
    bits = ~bits;
  }
  double value = 0;
  std::memcpy(&value, &bits, sizeof(double));
  return value;
}

void EgashinKRadixBatcherSortMPI::RadixSort(std::vector<double> &arr) {
  if (arr.size() <= 1) {
    return;
  }

  const int bits_per_pass = 8;
  const int num_buckets = 256;
  const int num_passes = static_cast<int>(sizeof(uint64_t) * 8 / bits_per_pass);

  std::size_t n = arr.size();
  std::vector<uint64_t> keys(n);
  std::vector<uint64_t> temp_keys(n);
  std::vector<double> temp_values(n);

  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = DoubleToSortable(arr[i]);
  }

  for (int pass = 0; pass < num_passes; ++pass) {
    int shift = pass * bits_per_pass;

    std::vector<std::size_t> count(num_buckets + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t digit = (keys[i] >> shift) & 0xFF;
      count[digit + 1]++;
    }

    for (int i = 0; i < num_buckets; ++i) {
      count[i + 1] += count[i];
    }

    for (std::size_t i = 0; i < n; ++i) {
      std::size_t digit = (keys[i] >> shift) & 0xFF;
      std::size_t pos = count[digit]++;
      temp_keys[pos] = keys[i];
      temp_values[pos] = arr[i];
    }

    std::swap(keys, temp_keys);
    std::swap(arr, temp_values);
  }

  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = SortableToDouble(keys[i]);
  }
}

void EgashinKRadixBatcherSortMPI::CompareExchange(std::vector<double> &arr, int i, int j) {
  auto sz = static_cast<int>(arr.size());
  if (j < sz && i < sz && arr[i] > arr[j]) {
    std::swap(arr[i], arr[j]);
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
void EgashinKRadixBatcherSortMPI::BatcherOddEvenMerge(std::vector<double> &arr, int lo, int n, int r) {
  int m = r * 2;
  if (m < n) {
    BatcherOddEvenMerge(arr, lo, n, m);
    BatcherOddEvenMerge(arr, lo + r, n, m);
    for (int i = lo + r; i + r < lo + n; i += m) {
      CompareExchange(arr, i, i + r);
    }
  } else {
    CompareExchange(arr, lo, lo + r);
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
void EgashinKRadixBatcherSortMPI::BatcherOddEvenMergeSort(std::vector<double> &arr, int lo, int n) {
  if (n > 1) {
    int m = n / 2;
    BatcherOddEvenMergeSort(arr, lo, m);
    BatcherOddEvenMergeSort(arr, lo + m, n - m);
    BatcherOddEvenMerge(arr, lo, n, 1);
  }
}

std::vector<std::pair<int, int>> EgashinKRadixBatcherSortMPI::GenerateBatcherNetwork(int n) {
  std::vector<std::pair<int, int>> comparators;
  if (n <= 1) {
    return comparators;
  }

  // Use odd-even transposition sort network
  // For uneven data distribution, need more phases to ensure convergence
  int num_phases = 2 * n;
  for (int phase = 0; phase < num_phases; ++phase) {
    if (phase % 2 == 0) {
      // Even phase: compare (0,1), (2,3), (4,5), ...
      for (int i = 0; i + 1 < n; i += 2) {
        comparators.emplace_back(i, i + 1);
      }
    } else {
      // Odd phase: compare (1,2), (3,4), (5,6), ...
      for (int i = 1; i + 1 < n; i += 2) {
        comparators.emplace_back(i, i + 1);
      }
    }
  }

  return comparators;
}

void EgashinKRadixBatcherSortMPI::MergeWithPartner(std::vector<double> &local_data, int partner_rank, int /*rank*/,
                                                   bool keep_lower) {
  int local_size = static_cast<int>(local_data.size());
  int partner_size = 0;

  // Exchange sizes
  MPI_Sendrecv(&local_size, 1, MPI_INT, partner_rank, 0, &partner_size, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  // Both processes must participate in data exchange even if one has empty data
  std::vector<double> partner_data(partner_size);

  // Exchange data (even if one side is empty)
  MPI_Sendrecv(local_data.data(), local_size, MPI_DOUBLE, partner_rank, 1, partner_data.data(), partner_size,
               MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // If both have no data, nothing to do
  if (local_size == 0 && partner_size == 0) {
    return;
  }

  // Merge and keep appropriate half
  std::vector<double> merged;
  merged.reserve(static_cast<std::size_t>(local_size) + static_cast<std::size_t>(partner_size));

  std::size_t i = 0;
  std::size_t j = 0;
  while (i < local_data.size() && j < partner_data.size()) {
    if (local_data[i] <= partner_data[j]) {
      merged.push_back(local_data[i++]);
    } else {
      merged.push_back(partner_data[j++]);
    }
  }
  while (i < local_data.size()) {
    merged.push_back(local_data[i++]);
  }
  while (j < partner_data.size()) {
    merged.push_back(partner_data[j++]);
  }

  // Keep lower or upper half based on rank comparison
  if (keep_lower) {
    local_data.assign(merged.begin(), merged.begin() + local_size);
  } else {
    local_data.assign(merged.end() - local_size, merged.end());
  }
}

void EgashinKRadixBatcherSortMPI::DistributeData(int total_size, int world_size, int rank, std::vector<double> &data,
                                                 std::vector<int> &counts, std::vector<int> &displs,
                                                 std::vector<double> &local_data) {
  int base_count = total_size / world_size;
  int remainder = total_size % world_size;

  for (int i = 0; i < world_size; ++i) {
    counts[i] = base_count + ((i < remainder) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : (displs[i - 1] + counts[i - 1]);
  }

  local_data.resize(counts[rank]);
  MPI_Scatterv(data.data(), counts.data(), displs.data(), MPI_DOUBLE, local_data.data(), counts[rank], MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
}

void EgashinKRadixBatcherSortMPI::PerformBatcherMerge(std::vector<double> &local_data, int world_size, int rank) {
  auto comparators = GenerateBatcherNetwork(world_size);

  for (const auto &[proc1, proc2] : comparators) {
    if (rank == proc1) {
      MergeWithPartner(local_data, proc2, rank, true);
    } else if (rank == proc2) {
      MergeWithPartner(local_data, proc1, rank, false);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void EgashinKRadixBatcherSortMPI::GatherResults(std::vector<double> &local_data, int total_size, int world_size,
                                                int rank, std::vector<double> &sorted_data) {
  if (rank == 0) {
    sorted_data.resize(total_size);
  }

  int new_local_size = static_cast<int>(local_data.size());
  std::vector<int> new_counts(world_size);
  MPI_Gather(&new_local_size, 1, MPI_INT, new_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> new_displs(world_size, 0);
  if (rank == 0) {
    for (int i = 1; i < world_size; ++i) {
      new_displs[i] = new_displs[i - 1] + new_counts[i - 1];
    }
  }

  MPI_Gatherv(local_data.data(), new_local_size, MPI_DOUBLE, sorted_data.data(), new_counts.data(), new_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool EgashinKRadixBatcherSortMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_size = 0;
  std::vector<double> data;

  if (rank == 0) {
    data = GetInput();
    total_size = static_cast<int>(data.size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    if (rank == 0) {
      GetOutput() = {};
    }
    return true;
  }

  std::vector<int> counts(world_size);
  std::vector<int> displs(world_size);
  std::vector<double> local_data;

  EgashinKRadixBatcherSortMPI::DistributeData(total_size, world_size, rank, data, counts, displs, local_data);

  RadixSort(local_data);

  EgashinKRadixBatcherSortMPI::PerformBatcherMerge(local_data, world_size, rank);

  std::vector<double> sorted_data;
  EgashinKRadixBatcherSortMPI::GatherResults(local_data, total_size, world_size, rank, sorted_data);

  if (rank == 0) {
    GetOutput() = sorted_data;
  }

  return true;
}

}  // namespace egashin_k_radix_batcher_sort
