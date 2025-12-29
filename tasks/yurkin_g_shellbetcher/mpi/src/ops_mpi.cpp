#include "yurkin_g_shellbetcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "yurkin_g_shellbetcher/common/include/common.hpp"

namespace yurkin_g_shellbetcher {
namespace {

template <typename It1, typename It2, typename OutIt>
void MergeSortedRanges(It1 first1, It1 last1, It2 first2, It2 last2, OutIt out) {
  while (first1 != last1 && first2 != last2) {
    if (*first1 <= *first2) {
      *out++ = *first1++;
    } else {
      *out++ = *first2++;
    }
  }
  while (first1 != last1) {
    *out++ = *first1++;
  }
  while (first2 != last2) {
    *out++ = *first2++;
  }
}

void ShellSort(std::vector<int> &a) {
  std::size_t n = a.size();
  std::size_t gap = 1;
  while (gap < n / 3) {
    gap = (gap * 3) + 1;
  }
  while (gap > 0) {
    for (std::size_t i = gap; i < n; ++i) {
      int tmp = a[i];
      std::size_t j = i;
      while (j >= gap && a[j - gap] > tmp) {
        a[j] = a[j - gap];
        j -= gap;
      }
      a[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}

void OddEvenBatcherMergeLocal(const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &out) {
  out.resize(a.size() + b.size());
  MergeSortedRanges(a.begin(), a.end(), b.begin(), b.end(), out.begin());
  for (int phase = 0; phase < 2; ++phase) {
    auto start = static_cast<std::size_t>(phase);
    for (std::size_t i = start; i + 1 < out.size(); i += 2) {
      if (out[i] > out[i + 1]) {
        std::swap(out[i], out[i + 1]);
      }
    }
  }
}

int ComputeNeighbor(int rank, int phase, int size) {
  int neighbor = MPI_PROC_NULL;
  if (phase % 2 == 0) {
    neighbor = (rank % 2 == 0) ? rank + 1 : rank - 1;
  } else {
    neighbor = (rank % 2 == 0) ? rank - 1 : rank + 1;
  }
  if (neighbor < 0 || neighbor >= size) {
    neighbor = MPI_PROC_NULL;
  }
  return neighbor;
}

void ExchangeAndMergeWithNeighbor(std::vector<int> &local_data, int neighbor) {
  int send_count = static_cast<int>(local_data.size());
  int recv_count = 0;
  MPI_Sendrecv(&send_count, 1, MPI_INT, neighbor, 10, &recv_count, 1, MPI_INT, neighbor, 10, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  std::vector<int> recv_buf(static_cast<std::size_t>(recv_count));
  MPI_Sendrecv(local_data.data(), send_count, MPI_INT, neighbor, 11, recv_buf.data(), recv_count, MPI_INT, neighbor, 11,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::vector<int> merged;
  OddEvenBatcherMergeLocal(local_data, recv_buf, merged);
  if (static_cast<int>(merged.size()) < send_count) {
    local_data.swap(merged);
    return;
  }
  if (neighbor == MPI_PROC_NULL) {
    local_data.swap(merged);
    return;
  }
  int my_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (my_rank < neighbor) {
    merged.resize(static_cast<std::size_t>(send_count));
    local_data.swap(merged);
  } else {
    std::vector<int> tmp;
    auto start_it = merged.begin() + static_cast<std::vector<int>::difference_type>(
                                         merged.size() - static_cast<std::size_t>(send_count));
    tmp.assign(start_it, merged.end());
    local_data.swap(tmp);
  }
}

void DoPowerOfTwoMergeStep(std::vector<int> &local_data, int rank, int size, int stages) {
  for (int stage = 0; stage < stages; ++stage) {
    for (int sub = stage; sub >= 0; --sub) {
      int partner_distance = 1 << sub;
      int partner = rank ^ partner_distance;
      if (partner < 0 || partner >= size) {
        continue;
      }
      int send_count = static_cast<int>(local_data.size());
      int recv_count = 0;
      MPI_Sendrecv(&send_count, 1, MPI_INT, partner, 0, &recv_count, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
      std::vector<int> recv_buf(static_cast<std::size_t>(recv_count));
      MPI_Sendrecv(local_data.data(), send_count, MPI_INT, partner, 1, recv_buf.data(), recv_count, MPI_INT, partner, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<int> merged;
      OddEvenBatcherMergeLocal(local_data, recv_buf, merged);
      if (static_cast<int>(merged.size()) < send_count) {
        local_data.swap(merged);
      } else {
        if (rank < partner) {
          merged.resize(static_cast<std::size_t>(send_count));
          local_data.swap(merged);
        } else {
          std::vector<int> tmp;
          auto start_it = merged.begin() + static_cast<std::vector<int>::difference_type>(
                                               merged.size() - static_cast<std::size_t>(send_count));
          tmp.assign(start_it, merged.end());
          local_data.swap(tmp);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

void DoOddEvenTransposition(std::vector<int> &local_data, int rank, int size) {
  for (int phase = 0; phase < size; ++phase) {
    int neighbor = ComputeNeighbor(rank, phase, size);
    if (neighbor != MPI_PROC_NULL) {
      ExchangeAndMergeWithNeighbor(local_data, neighbor);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

}  // namespace

YurkinGShellBetcherMPI::YurkinGShellBetcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGShellBetcherMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool YurkinGShellBetcherMPI::PreProcessingImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return false;
  }
  return GetInput() > 0;
}

bool YurkinGShellBetcherMPI::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    return false;
  }
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  InType base = n / size;
  InType rem = n % size;
  InType local_n = base + (rank < rem ? 1 : 0);
  std::vector<int> local_data;
  local_data.reserve(static_cast<std::size_t>(local_n));
  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);
  InType offset = (base * rank) + std::min<InType>(rank, rem);
  for (InType i = 0; i < offset; ++i) {
    (void)dist(rng);
  }
  for (InType i = 0; i < local_n; ++i) {
    local_data.push_back(dist(rng));
  }
  std::vector<int> all_counts;
  all_counts.resize(size);
  int send_count = static_cast<int>(local_data.size());
  MPI_Gather(&send_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> displs;
  std::vector<int> recvbuf;
  if (rank == 0) {
    displs.resize(size);
    int total = 0;
    for (int i = 0; i < size; ++i) {
      displs[i] = total;
      total += all_counts[i];
    }
    recvbuf.resize(static_cast<std::size_t>(total));
  }
  MPI_Gatherv(local_data.data(), send_count, MPI_INT, rank == 0 ? recvbuf.data() : nullptr,
              rank == 0 ? all_counts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_INT, 0, MPI_COMM_WORLD);
  std::int64_t global_checksum = 0;
  if (rank == 0) {
    ShellSort(recvbuf);
    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> merged;
    auto mid = recvbuf.size() / 2;
    left.assign(recvbuf.begin(), recvbuf.begin() + static_cast<std::vector<int>::difference_type>(mid));
    right.assign(recvbuf.begin() + static_cast<std::vector<int>::difference_type>(mid), recvbuf.end());
    OddEvenBatcherMergeLocal(left, right, merged);
    ShellSort(merged);
    for (int v : merged) {
      global_checksum += static_cast<std::int64_t>(v);
    }
  }
  MPI_Bcast(&global_checksum, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  GetOutput() = static_cast<OutType>(global_checksum & 0x7FFFFFFF);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool YurkinGShellBetcherMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace yurkin_g_shellbetcher
