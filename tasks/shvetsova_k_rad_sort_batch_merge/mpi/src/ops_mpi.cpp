#include "shvetsova_k_rad_sort_batch_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

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

  // ---- Распределение данных (каждый процесс считает сам) ----
  std::vector<int> counts(proc_count);
  std::vector<int> displs(proc_count);
  int base = size / proc_count;
  int rem = size % proc_count;
  int offset = 0;
  for (int i = 0; i < proc_count; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }

  int local_size = counts[rank];
  std::vector<double> local(local_size);

  MPI_Scatterv(rank == 0 ? data_.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, local.data(), local_size,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // ---- 1. Локальная поразрядная сортировка ----
  RadixSortLocal(local);

  // ---- 2. Глобальное слияние (Odd-Even Transposition Sort) ----
  for (int step = 0; step < proc_count; ++step) {
    int partner;
    if (step % 2 == 0) {  // Четный шаг
      partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
    } else {  // Нечетный шаг
      partner = (rank % 2 == 0) ? rank - 1 : rank + 1;
    }

    if (partner >= 0 && partner < proc_count) {
      int partner_size = counts[partner];
      std::vector<double> recv(partner_size);

      MPI_Sendrecv(local.data(), local_size, MPI_DOUBLE, partner, 0, recv.data(), partner_size, MPI_DOUBLE, partner, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<double> merged(local_size + partner_size);
      std::merge(local.begin(), local.end(), recv.begin(), recv.end(), merged.begin());

      if (rank < partner) {
        std::copy(merged.begin(), merged.begin() + local_size, local.begin());
      } else {
        std::copy(merged.end() - local_size, merged.end(), local.begin());
      }
    }
  }

  // ---- 3. Сбор результата ----
  // Важно: все процессы ресайзят свой вектор data_ перед получением/рассылкой данных
  data_.resize(size);

  MPI_Gatherv(local.data(), local_size, MPI_DOUBLE, rank == 0 ? data_.data() : nullptr, counts.data(), displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // РЕШЕНИЕ ПРОБЛЕМЫ ТЕСТОВ: рассылаем финальный результат всем процессам
  // Теперь CheckTestOutputData увидит отсортированные данные на всех рангах
  MPI_Bcast(data_.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Теперь GetOutput() на каждом процессе содержит правильный ответ
  GetOutput() = data_;

  return true;
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
    int count[base] = {0};

    for (double x : vec) {
      int digit = (static_cast<int>(std::abs(x)) / exp) % base;
      count[digit]++;
    }

    for (int i = 1; i < base; ++i) {
      count[i] += count[i - 1];
    }

    for (int i = static_cast<int>(vec.size()) - 1; i >= 0; --i) {
      int digit = (static_cast<int>(std::abs(vec[i])) / exp) % base;
      output[--count[digit]] = vec[i];
    }
    vec = std::move(output);
  }
}

bool ShvetsovaKRadSortBatchMergeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shvetsova_k_rad_sort_batch_merge
