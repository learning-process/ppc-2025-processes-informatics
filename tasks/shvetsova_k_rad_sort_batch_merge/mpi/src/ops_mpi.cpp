#include "shvetsova_k_rad_sort_batch_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>
#include <utility>
#include <vector>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

ShvetsovaKRadSortBatchMergeMPI::ShvetsovaKRadSortBatchMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ShvetsovaKRadSortBatchMergeMPI::ValidationImpl() {
  return true;
}

bool ShvetsovaKRadSortBatchMergeMPI::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool ShvetsovaKRadSortBatchMergeMPI::RunImpl() {
  int proc_count = 0;
  int rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size = (rank == 0) ? static_cast<int>(data_.size()) : 0;
  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (size == 0) {
    if (rank == 0) {
      GetOutput() = OutType{};
    }
    return true;
  }

  std::vector<int> counts(proc_count);
  std::vector<int> displs(proc_count);
  CreateDistribution(proc_count, size, counts, displs);

  std::vector<double> local(counts.at(rank));
  ScatterData(data_, local, counts, displs, rank);

  RadixSortLocal(local);
  OddEvenMerge(local, counts, rank, proc_count);

  data_.resize(size);
  GatherAndBroadcast(data_, local, counts, displs, rank);

  GetOutput() = data_;
  return true;
}

void ShvetsovaKRadSortBatchMergeMPI::CreateDistribution(int proc_count, int size, std::vector<int> &counts,
                                                        std::vector<int> &displs) {
  int base = size / proc_count;
  int rem = size % proc_count;
  int offset = 0;

  for (int i = 0; i < proc_count; ++i) {
    counts.at(i) = base + (i < rem ? 1 : 0);
    displs.at(i) = offset;
    offset += counts.at(i);
  }
}

void ShvetsovaKRadSortBatchMergeMPI::ScatterData(const std::vector<double> &data, std::vector<double> &local,
                                                 const std::vector<int> &counts, const std::vector<int> &displs,
                                                 int rank) {
  MPI_Scatterv(rank == 0 ? data.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, local.data(),
               static_cast<int>(local.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int ShvetsovaKRadSortBatchMergeMPI::CalculatePartner(int step, int rank, int proc_count) {
  int partner = -1;
  if (step % 2 == 0) {
    partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
  } else {
    partner = (rank % 2 == 0) ? rank - 1 : rank + 1;
  }

  if (partner < 0 || partner >= proc_count) {
    return -1;
  }
  return partner;
}

void ShvetsovaKRadSortBatchMergeMPI::PerformMerge(std::vector<double> &local, const std::vector<double> &recv, int rank,
                                                  int partner) {
  int local_size = static_cast<int>(local.size());
  std::vector<double> merged(local_size + recv.size());

  std::ranges::merge(local, recv, merged.begin());

  if (rank < partner) {
    std::ranges::copy(merged | std::views::take(local_size), local.begin());
  } else {
    std::ranges::copy(merged | std::views::drop(merged.size() - local_size), local.begin());
  }
}

void ShvetsovaKRadSortBatchMergeMPI::OddEvenMerge(std::vector<double> &local, const std::vector<int> &counts, int rank,
                                                  int proc_count) {
  for (int step = 0; step < proc_count; ++step) {
    int partner = CalculatePartner(step, rank, proc_count);

    if (partner == -1) {
      continue;
    }

    int partner_size = counts.at(partner);
    std::vector<double> recv(partner_size);

    MPI_Sendrecv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, partner, 0, recv.data(), partner_size,
                 MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    PerformMerge(local, recv, rank, partner);
  }
}

void ShvetsovaKRadSortBatchMergeMPI::GatherAndBroadcast(std::vector<double> &data, const std::vector<double> &local,
                                                        const std::vector<int> &counts, const std::vector<int> &displs,
                                                        int rank) {
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, rank == 0 ? data.data() : nullptr,
              counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void ShvetsovaKRadSortBatchMergeMPI::RadixSortLocal(std::vector<double> &vec) {
  if (vec.empty()) {
    return;
  }

  int max_val = 0;
  for (double x : vec) {
    max_val = std::max(max_val, static_cast<int>(std::abs(x)));
  }

  const int base = 10;
  for (int exp = 1; max_val / exp > 0; exp *= base) {
    std::vector<double> output(vec.size());
    std::array<int, base> count{};

    for (double x : vec) {
      int digit = (static_cast<int>(std::abs(x)) / exp) % base;
      count.at(digit)++;
    }

    for (int i = 1; i < base; ++i) {
      count.at(i) += count.at(i - 1);
    }

    for (int i = static_cast<int>(vec.size()) - 1; i >= 0; --i) {
      int digit = (static_cast<int>(std::abs(vec.at(i))) / exp) % base;
      output.at(--count.at(digit)) = vec.at(i);
    }

    vec = std::move(output);
  }
}

bool ShvetsovaKRadSortBatchMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_rad_sort_batch_merge
