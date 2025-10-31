#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"

namespace sizov_d_string_mismatch_count {

SizovDStringMismatchCountMPI::SizovDStringMismatchCountMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput() = 0;
}

bool SizovDStringMismatchCountMPI::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);
  return !a.empty() && a.size() == b.size();
}

bool SizovDStringMismatchCountMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  str_a_ = std::get<0>(input);
  str_b_ = std::get<1>(input);
  return true;
}

bool SizovDStringMismatchCountMPI::RunImpl() {
  // --- Базовая инфа ---
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int pid = static_cast<int>(getpid());
  const int total_size = static_cast<int>(str_a_.size());

  std::cerr << "[Rank " << rank << " | pid " << pid << "] Start RunImpl (MPI): size=" << size << " len=" << total_size
            << std::endl;

  // --- Барьер перед измерением ---
  MPI_Barrier(MPI_COMM_WORLD);
  auto start_time = std::chrono::high_resolution_clock::now();

  // --- Локная работа ---
  int local_result = 0;
  for (int i = rank; i < total_size; i += size) {
    if (str_a_[i] != str_b_[i]) {
      ++local_result;
    }
  }

  std::cerr << "[Rank " << rank << "] Local mismatches = " << local_result << std::endl;

  // --- Глобальная редукция ---
  int global_result = 0;
  MPI_Barrier(MPI_COMM_WORLD);  // синхронизация перед Reduce
  MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // --- Завершение ---
  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

  std::cerr << "[Rank " << rank << " | pid " << pid << "] End RunImpl → local=" << local_result
            << ", global=" << global_result << ", time=" << elapsed_ms << " ms" << std::endl;

  GetOutput() = global_result;
  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_string_mismatch_count
