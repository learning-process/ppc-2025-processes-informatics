#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <unistd.h>  // для getpid()

#include <chrono>
#include <iostream>

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
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  bool mpi_ready = (mpi_initialized != 0);

  const int total_size = static_cast<int>(str_a_.size());
  const int pid = static_cast<int>(getpid());

  if (!mpi_ready) {
    std::cerr << "[SEQ][pid " << pid << "] MPI not initialized → running sequentially\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    int local_result = 0;
    for (int i = 0; i < total_size; ++i) {
      if (str_a_[i] != str_b_[i]) {
        ++local_result;
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    double elapsed_ms = static_cast<double>(micros) / 1000.0;

    std::cerr << "[SEQ][pid " << pid << "] Completed → mismatches=" << local_result << ", time=" << elapsed_ms
              << " ms\n";
    GetOutput() = local_result;
    return true;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cerr << "[Rank " << rank << " | pid " << pid << "] Start RunImpl (MPI): size=" << size << " len=" << total_size
            << "\n";

  // Барьер перед работой
  MPI_Barrier(MPI_COMM_WORLD);

  auto start_time = std::chrono::high_resolution_clock::now();

  int local_result = 0;
  for (int i = rank; i < total_size; i += size) {
    if (str_a_[i] != str_b_[i]) {
      ++local_result;
    }
  }

  std::cerr << "[Rank " << rank << "] Local mismatches=" << local_result << "\n";

  int global_result = 0;
  auto sync_start = std::chrono::high_resolution_clock::now();

  // Барьер перед Reduce
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_micros = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  auto sync_micros = std::chrono::duration_cast<std::chrono::microseconds>(end_time - sync_start).count();

  double total_ms = static_cast<double>(total_micros) / 1000.0;
  double sync_ms = static_cast<double>(sync_micros) / 1000.0;

  std::cerr << "[Rank " << rank << " | pid " << pid << "] End RunImpl → local=" << local_result
            << ", global=" << global_result << " | total=" << total_ms << " ms (sync=" << sync_ms << " ms)\n";

  GetOutput() = global_result;
  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_string_mismatch_count
