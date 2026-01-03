#include "shvetsova_k_rad_sort_batch_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
  BatcherOddEvenMergeParallel(local, counts, rank, proc_count);

  data_.resize(static_cast<std::size_t>(size));
  GatherAndBroadcast(data_, local, counts, displs, rank);

  GetOutput() = data_;
  return true;
}

// распределение

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

void ShvetsovaKRadSortBatchMergeMPI::BatcherOddEvenMergeSequential(std::vector<double> &a, std::vector<double> &b,
                                                                   std::vector<double> &result) {
  result.resize(a.size() + b.size());
  std::ranges::merge(a, b, result.begin());
}

// Бэтчер

int ShvetsovaKRadSortBatchMergeMPI::ComputePartner(int step, int rank, int proc_count) {
  if (step % 2 == 0) {
    return (rank % 2 == 0 && rank + 1 < proc_count) ? rank + 1 : (rank % 2 == 1) ? rank - 1 : -1;
  }
  return (rank % 2 == 0 && rank > 0) ? rank - 1 : (rank % 2 == 1 && rank + 1 < proc_count) ? rank + 1 : -1;
}

void ShvetsovaKRadSortBatchMergeMPI::BatcherOddEvenMergeParallel(std::vector<double> &local,
                                                                 const std::vector<int> &counts, int rank,
                                                                 int proc_count) {
  for (int step = 0; step < proc_count; ++step) {
    int partner = ComputePartner(step, rank, proc_count);

    if (partner >= 0 && partner < proc_count) {
      ExchangeAndMerge(local, counts, partner, rank < partner);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void ShvetsovaKRadSortBatchMergeMPI::ExchangeAndMerge(std::vector<double> &local, const std::vector<int> &counts,
                                                      int partner, bool keep_smaller) {
  int partner_size = counts.at(partner);
  std::vector<double> recv(partner_size);

  MPI_Sendrecv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, partner, 0, recv.data(), partner_size,
               MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<double> merged;
  BatcherOddEvenMergeSequential(local, recv, merged);

  if (keep_smaller) {
    std::ranges::copy(merged | std::views::take(local.size()), local.begin());
  } else {
    std::ranges::copy(merged | std::views::drop(merged.size() - local.size()), local.begin());
  }
}

void ShvetsovaKRadSortBatchMergeMPI::GatherAndBroadcast(std::vector<double> &data, const std::vector<double> &local,
                                                        const std::vector<int> &counts, const std::vector<int> &displs,
                                                        int rank) {
  MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, rank == 0 ? data.data() : nullptr,
              counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(data.data(), static_cast<int>(data.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// поразрядная сортировка

static std::int64_t Abs64(std::int64_t x) {
  return std::llabs(x);
}

static int GetDigit(std::int64_t x, std::int64_t exp, int base) {
  int digit = static_cast<int>((Abs64(x) / exp) % base);
  return (x < 0) ? (base - 1 - digit) : digit;
}

void ShvetsovaKRadSortBatchMergeMPI::RadixSortLocal(std::vector<double> &vec) {
  if (vec.empty()) {
    return;
  }

  std::vector<std::int64_t> values(vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i) {
    values[i] = static_cast<std::int64_t>(vec[i]);
  }

  std::int64_t max_val = 0;
  for (std::int64_t x : values) {
    max_val = std::max(max_val, Abs64(x));
  }

  constexpr int base = 256;

  for (std::int64_t exp = 1; max_val / exp > 0; exp *= base) {
    std::vector<std::int64_t> output(values.size());
    std::array<int, base> count{};
    count.fill(0);

    for (std::int64_t x : values) {
      ++count.at(GetDigit(x, exp, base));
    }

    for (int i = 1; i < base; ++i) {
      count.at(i) += count.at(i - 1);
    }

    for (std::size_t i = values.size(); i-- > 0;) {
      std::int64_t x = values[i];
      int d = GetDigit(x, exp, base);
      output.at(static_cast<std::size_t>(--count.at(d))) = x;
    }

    values = std::move(output);
  }

  for (std::size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<double>(values[i]);
  }
}

bool ShvetsovaKRadSortBatchMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_rad_sort_batch_merge
