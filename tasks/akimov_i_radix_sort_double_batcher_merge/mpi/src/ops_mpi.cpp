#include "akimov_i_radix_sort_double_batcher_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "akimov_i_radix_sort_double_batcher_merge/common/include/common.hpp"

namespace akimov_i_radix_sort_double_batcher_merge {

AkimovIRadixBatcherSortMPI::AkimovIRadixBatcherSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool AkimovIRadixBatcherSortMPI::ValidationImpl() {
  return true;
}
bool AkimovIRadixBatcherSortMPI::PreProcessingImpl() {
  return true;
}
bool AkimovIRadixBatcherSortMPI::PostProcessingImpl() {
  return true;
}

uint64_t AkimovIRadixBatcherSortMPI::packDouble(double v) noexcept {
  uint64_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  // transform IEEE-754 double to lexicographically sortable integer
  if (bits & (1ULL << 63)) {  // negative numbers
    bits = ~bits;
  } else {  // positive numbers
    bits ^= (1ULL << 63);
  }
  return bits;
}

double AkimovIRadixBatcherSortMPI::unpackDouble(uint64_t k) noexcept {
  if (k & (1ULL << 63)) {
    k ^= (1ULL << 63);
  } else {
    k = ~k;
  }
  double v;
  std::memcpy(&v, &k, sizeof(v));
  return v;
}

void AkimovIRadixBatcherSortMPI::lsdRadixSort(std::vector<double> &arr) {
  const std::size_t n = arr.size();
  if (n < 2) {
    return;
  }

  constexpr int BITS = 8;
  constexpr int BUCKETS = 1 << BITS;
  constexpr int PASSES = (int)((sizeof(uint64_t) * 8) / BITS);

  std::vector<uint64_t> keys(n), tmpK(n);
  std::vector<double> tmpV(n);

  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = packDouble(arr[i]);
  }

  for (int pass = 0; pass < PASSES; ++pass) {
    int shift = pass * BITS;
    std::vector<std::size_t> cnt(BUCKETS + 1, 0);

    // counting
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t digit = (keys[i] >> shift) & (BUCKETS - 1);
      ++cnt[digit + 1];
    }
    // prefix sum
    for (int i = 0; i < BUCKETS; ++i) {
      cnt[i + 1] += cnt[i];
    }

    // place
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t digit = (keys[i] >> shift) & (BUCKETS - 1);
      std::size_t pos = cnt[digit]++;
      tmpK[pos] = keys[i];
      tmpV[pos] = arr[i];
    }

    keys.swap(tmpK);
    arr.swap(tmpV);
  }

  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = unpackDouble(keys[i]);
  }
}

void AkimovIRadixBatcherSortMPI::cmpSwap(std::vector<double> &arr, int i, int j) noexcept {
  int sz = static_cast<int>(arr.size());
  if (i < sz && j < sz && arr[i] > arr[j]) {
    std::swap(arr[i], arr[j]);
  }
}

// recursive odd-even merge (Batcher-like)
void AkimovIRadixBatcherSortMPI::oddEvenMergeRec(std::vector<double> &arr, int start, int len, int stride) {
  int step = stride * 2;
  if (step < len) {
    oddEvenMergeRec(arr, start, len, step);
    oddEvenMergeRec(arr, start + stride, len, step);
    for (int i = start + stride; i + stride < start + len; i += step) {
      cmpSwap(arr, i, i + stride);
    }
  } else {
    cmpSwap(arr, start, start + stride);
  }
}

void AkimovIRadixBatcherSortMPI::oddEvenMergeSortRec(std::vector<double> &arr, int start, int len) {
  if (len > 1) {
    int mid = len / 2;
    oddEvenMergeSortRec(arr, start, mid);
    oddEvenMergeSortRec(arr, start + mid, len - mid);
    oddEvenMergeRec(arr, start, len, 1);
  }
}

std::vector<std::pair<int, int>> AkimovIRadixBatcherSortMPI::buildOddEvenPhasePairs(int procs) {
  std::vector<std::pair<int, int>> phases;
  if (procs <= 1) {
    return phases;
  }

  // build pairs for several odd-even phases to cover uneven distributions
  int phases_count = procs * 2;
  for (int ph = 0; ph < phases_count; ++ph) {
    if ((ph & 1) == 0) {
      for (int i = 0; i + 1 < procs; i += 2) {
        phases.emplace_back(i, i + 1);
      }
    } else {
      for (int i = 1; i + 1 < procs; i += 2) {
        phases.emplace_back(i, i + 1);
      }
    }
  }
  return phases;
}

void AkimovIRadixBatcherSortMPI::exchangeAndSelect(std::vector<double> &local, int partner, int /*rank*/,
                                                   bool keep_lower) {
  int my_cnt = static_cast<int>(local.size());
  int partner_cnt = 0;

  MPI_Sendrecv(&my_cnt, 1, MPI_INT, partner, 0, &partner_cnt, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<double> partner_buf(partner_cnt);
  // Even if counts are zero, MPI_Sendrecv must be called with valid pointers (size 0 is ok).
  MPI_Sendrecv(local.data(), my_cnt, MPI_DOUBLE, partner, 1, partner_buf.data(), partner_cnt, MPI_DOUBLE, partner, 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (my_cnt == 0 && partner_cnt == 0) {
    return;
  }

  // merge two sorted vectors
  std::vector<double> merged;
  merged.reserve(static_cast<std::size_t>(my_cnt + partner_cnt));
  std::size_t i = 0, j = 0;
  while (i < local.size() && j < partner_buf.size()) {
    if (local[i] <= partner_buf[j]) {
      merged.push_back(local[i++]);
    } else {
      merged.push_back(partner_buf[j++]);
    }
  }
  while (i < local.size()) {
    merged.push_back(local[i++]);
  }
  while (j < partner_buf.size()) {
    merged.push_back(partner_buf[j++]);
  }

  if (keep_lower) {
    local.assign(merged.begin(), merged.begin() + my_cnt);
  } else {
    local.assign(merged.end() - my_cnt, merged.end());
  }
}

void AkimovIRadixBatcherSortMPI::computeCountsDispls(int total, int world, std::vector<int> &counts,
                                                     std::vector<int> &displs) {
  int base = total / world;
  int rem = total % world;
  for (int i = 0; i < world; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }
}

void AkimovIRadixBatcherSortMPI::scatterData(int total, int world, int rank, std::vector<double> &data,
                                             std::vector<int> &counts, std::vector<int> &displs,
                                             std::vector<double> &local) {
  computeCountsDispls(total, world, counts, displs);
  local.resize(counts[rank]);
  MPI_Scatterv(data.data(), counts.data(), displs.data(), MPI_DOUBLE, local.data(), counts[rank], MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
}

void AkimovIRadixBatcherSortMPI::performNetworkMerge(std::vector<double> &local, int world, int rank) {
  auto pairs = buildOddEvenPhasePairs(world);
  for (const auto &pr : pairs) {
    int a = pr.first, b = pr.second;
    if (rank == a) {
      exchangeAndSelect(local, b, rank, true);
    } else if (rank == b) {
      exchangeAndSelect(local, a, rank, false);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void AkimovIRadixBatcherSortMPI::gatherData(int total, int world, int rank, std::vector<double> &local,
                                            std::vector<double> &out) {
  if (rank == 0) {
    out.resize(total);
  }

  int my_sz = static_cast<int>(local.size());
  std::vector<int> sizes(world);
  MPI_Gather(&my_sz, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs(world, 0);
  if (rank == 0) {
    for (int i = 1; i < world; ++i) {
      displs[i] = displs[i - 1] + sizes[i - 1];
    }
  }

  MPI_Gatherv(local.data(), my_sz, MPI_DOUBLE, out.data(), sizes.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool AkimovIRadixBatcherSortMPI::RunImpl() {
  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  int total = 0;
  std::vector<double> all;
  if (rank == 0) {
    all = GetInput();
    total = static_cast<int>(all.size());
  }

  MPI_Bcast(&total, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total == 0) {
    if (rank == 0) {
      GetOutput() = {};
    }
    return true;
  }

  std::vector<int> counts(world), displs(world);
  std::vector<double> local;
  scatterData(total, world, rank, all, counts, displs, local);

  // local sort
  lsdRadixSort(local);

  // distributed merge using network of pair exchanges
  performNetworkMerge(local, world, rank);

  std::vector<double> result;
  gatherData(total, world, rank, local, result);

  if (rank == 0) {
    GetOutput() = result;
  }
  return true;
}

}  // namespace akimov_i_radix_sort_double_batcher_merge
