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

  // Используем Batcher's odd-even merge
  BatcherOddEvenMergeParallel(local, counts, rank, proc_count);

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

void ShvetsovaKRadSortBatchMergeMPI::BatcherOddEvenMergeSequential(std::vector<double> &arr1, std::vector<double> &arr2,
                                                                   std::vector<double> &result) {
  result.resize(arr1.size() + arr2.size());
  std::ranges::merge(arr1, arr2, result.begin());
}

// Batcher's odd-even merge
void ShvetsovaKRadSortBatchMergeMPI::BatcherOddEvenMergeParallel(std::vector<double> &local,
                                                                 const std::vector<int> &counts, int rank,
                                                                 int proc_count) {
  for (int step = 0; step < proc_count; ++step) {
    int partner = -1;

    if (step % 2 == 0) {
      // Четный шаг: пары (0-1, 2-3, ...)
      if (rank % 2 == 0 && rank + 1 < proc_count) {
        partner = rank + 1;
      } else if (rank % 2 == 1) {
        partner = rank - 1;
      }
    } else {
      // Нечетный шаг: пары (1-2, 3-4, ...)
      if (rank % 2 == 0 && rank > 0) {
        partner = rank - 1;
      } else if (rank % 2 == 1 && rank + 1 < proc_count) {
        partner = rank + 1;
      }
    }

    if (partner != -1 && partner >= 0 && partner < proc_count) {
      bool keep_smaller = (rank < partner);
      ExchangeAndMerge(local, counts, partner, keep_smaller);
    }

    // Синхронизация всех процессов
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void ShvetsovaKRadSortBatchMergeMPI::ExchangeAndMerge(std::vector<double> &local, const std::vector<int> &counts,
                                                      int partner, bool keep_smaller) {
  if (partner < 0 || partner >= static_cast<int>(counts.size())) {
    return;
  }

  int partner_size = counts.at(partner);
  std::vector<double> recv(partner_size);

  // Обмен данными
  MPI_Sendrecv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, partner, 0, recv.data(), partner_size,
               MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Слияние двух отсортированных массивов
  std::vector<double> merged;
  BatcherOddEvenMergeSequential(local, recv, merged);

  // Выбираем нужную половину
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

void ShvetsovaKRadSortBatchMergeMPI::RadixSortLocal(std::vector<double> &vec) {
  if (vec.empty()) {
    return;
  }

  // Для целых чисел, представленных как double
  std::vector<long long> int_vec(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    int_vec[i] = static_cast<long long>(vec[i]);
  }

  // Находим максимальное значение по модулю
  long long max_val = 0;
  for (long long x : int_vec) {
    long long abs_x = std::llabs(x);
    if (abs_x > max_val) {
      max_val = abs_x;
    }
  }

  // Поразрядная сортировка по основанию 256 (байтовая)
  const int base = 256;
  for (long long exp = 1; max_val / exp > 0; exp *= base) {
    std::vector<long long> output(int_vec.size());
    std::array<int, base> count{};

    // Подсчет цифр
    for (long long x : int_vec) {
      int digit = (std::llabs(x) / exp) % base;
      if (x < 0) {
        digit = base - 1 - digit;  // Инвертируем для отрицательных
      }
      count[digit]++;
    }

    // Префиксная сумма
    for (int i = 1; i < base; ++i) {
      count[i] += count[i - 1];
    }

    // Распределение
    for (int i = static_cast<int>(int_vec.size()) - 1; i >= 0; --i) {
      long long x = int_vec[i];
      int digit = (std::llabs(x) / exp) % base;
      if (x < 0) {
        digit = base - 1 - digit;
      }
      output[--count[digit]] = x;
    }

    int_vec = std::move(output);
  }

  // Копируем обратно в double
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<double>(int_vec[i]);
  }
}
bool ShvetsovaKRadSortBatchMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_rad_sort_batch_merge
