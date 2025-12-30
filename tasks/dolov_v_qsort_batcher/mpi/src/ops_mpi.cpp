#include "dolov_v_qsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "dolov_v_qsort_batcher/common/include/common.hpp"

namespace dolov_v_qsort_batcher {

DolovVQsortBatcherMPI::DolovVQsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DolovVQsortBatcherMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool DolovVQsortBatcherMPI::PreProcessingImpl() {
  return true;
}

bool DolovVQsortBatcherMPI::RunImpl() {
  int procCount = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procCount);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int totalSize = 0;
  if (rank == 0) {
    totalSize = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&totalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (totalSize == 0) {
    return true;
  }

  std::vector<int> counts(procCount);
  std::vector<int> offsets(procCount);
  setupWorkload(totalSize, procCount, counts, offsets);

  std::vector<double> localVec(counts[rank]);
  splitData(GetInput(), localVec, counts, offsets);

  if (!localVec.empty()) {
    applyQuicksort(localVec.data(), 0, static_cast<int>(localVec.size()) - 1);
  }

  runBatcherProcess(rank, procCount, localVec);

  std::vector<double> globalRes;
  if (rank == 0) {
    globalRes.resize(totalSize);
  }

  collectData(globalRes, localVec, totalSize, counts, offsets);

  if (rank == 0) {
    GetOutput() = std::move(globalRes);
  }

  return true;
}

void DolovVQsortBatcherMPI::setupWorkload(int totalSize, int procCount, std::vector<int> &counts,
                                          std::vector<int> &offsets) {
  int base = totalSize / procCount;
  int rem = totalSize % procCount;
  for (int i = 0; i < procCount; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    offsets[i] = (i == 0) ? 0 : offsets[i - 1] + counts[i - 1];
  }
}

void DolovVQsortBatcherMPI::splitData(const std::vector<double> &source, std::vector<double> &local,
                                      const std::vector<int> &counts, const std::vector<int> &offsets) {
  MPI_Scatterv(source.data(), counts.data(), offsets.data(), MPI_DOUBLE, local.data(), static_cast<int>(local.size()),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int DolovVQsortBatcherMPI::getHoarePartition(double *array, int low, int high) {
  double pivot = array[low + (high - low) / 2];
  int i = low - 1;
  int j = high + 1;

  while (true) {
    while (array[++i] < pivot);
    while (array[--j] > pivot);
    if (i >= j) {
      return j;
    }
    std::swap(array[i], array[j]);
  }
}

void DolovVQsortBatcherMPI::applyQuicksort(double *array, int low, int high) {
  if (low < high) {
    int p = getHoarePartition(array, low, high);
    applyQuicksort(array, low, p);
    applyQuicksort(array, p + 1, high);
  }
}

void DolovVQsortBatcherMPI::runBatcherProcess(int rank, int procCount, std::vector<double> &localVec) {
  for (int p = 1; p < procCount; p <<= 1) {
    for (int k = p; k >= 1; k >>= 1) {
      for (int j = k % p; j <= procCount - 1 - k; j += (k << 1)) {
        for (int i = 0; i < k; ++i) {
          int left = j + i;
          int right = j + i + k;
          if (left / (p << 1) == right / (p << 1)) {
            if (rank == left) {
              exchangeAndMerge(rank, right, localVec, true);
            } else if (rank == right) {
              exchangeAndMerge(rank, left, localVec, false);
            }
          }
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

void DolovVQsortBatcherMPI::exchangeAndMerge(int /*rank*/, int partner, std::vector<double> &localVec, bool keepSmall) {
  int localSize = static_cast<int>(localVec.size());
  int partnerSize = 0;

  MPI_Sendrecv(&localSize, 1, MPI_INT, partner, 0, &partnerSize, 1, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<double> remoteVec(partnerSize);
  MPI_Sendrecv(localVec.data(), localSize, MPI_DOUBLE, partner, 1, remoteVec.data(), partnerSize, MPI_DOUBLE, partner,
               1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<double> merged = mergeTwoSortedArrays(localVec, remoteVec);

  localVec.clear();
  if (keepSmall) {
    localVec.assign(merged.begin(), merged.begin() + localSize);
  } else {
    localVec.assign(merged.end() - localSize, merged.end());
  }
}

std::vector<double> DolovVQsortBatcherMPI::mergeTwoSortedArrays(const std::vector<double> &first,
                                                                const std::vector<double> &second) {
  std::vector<double> res;
  res.reserve(first.size() + second.size());
  size_t i = 0;
  size_t j = 0;
  while (i < first.size() && j < second.size()) {
    if (first[i] <= second[j]) {
      res.push_back(first[i++]);
    } else {
      res.push_back(second[j++]);
    }
  }
  while (i < first.size()) {
    res.push_back(first[i++]);
  }
  while (j < second.size()) {
    res.push_back(second[j++]);
  }
  return res;
}

void DolovVQsortBatcherMPI::collectData(std::vector<double> &globalRes, const std::vector<double> &local,
                                        int /*totalSize*/, const std::vector<int> &counts,
                                        const std::vector<int> &offsets) {
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, globalRes.data(), counts.data(), offsets.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool DolovVQsortBatcherMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dolov_v_qsort_batcher
