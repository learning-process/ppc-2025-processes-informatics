#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <chrono>
#include <iostream>
#include <thread>

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
  using namespace std::chrono;
  int rank = 0, size = 1;

  bool mpi_ready = false;
  try {
    int initialized = 0;
    MPI_Initialized(&initialized);
    mpi_ready = (initialized != 0);
  } catch (...) {
    mpi_ready = false;
  }

  const int total_size = static_cast<int>(str_a_.size());

  // ---------- SEQ fallback ----------
  if (!mpi_ready) {
    std::cerr << "[MPI][fallback][thread " << std::this_thread::get_id()
              << "] MPI not initialized, running sequentially\n";

    auto start_time = high_resolution_clock::now();
    int local_result = 0;
    for (int i = 0; i < total_size; ++i) {
      if (str_a_[i] != str_b_[i]) {
        ++local_result;
      }
    }
    auto end_time = high_resolution_clock::now();
    double elapsed_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

    GetOutput() = local_result;
    std::cerr << "[MPI][fallback] Done, result=" << local_result << " | time=" << elapsed_ms << " ms\n";
    return true;
  }

  // ---------- MPI mode ----------
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cerr << "[Rank " << rank << "] Start RunImpl (MPI): size=" << size << " len=" << total_size << "\n";

  auto start_time = high_resolution_clock::now();

  int local_result = 0;
  for (int i = rank; i < total_size; i += size) {
    if (str_a_[i] != str_b_[i]) {
      ++local_result;
    }
  }

  int global_result = 0;
  MPI_Reduce(&local_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto end_time = high_resolution_clock::now();
  double elapsed_ms = duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

  GetOutput() = global_result;

  std::cerr << "[Rank " << rank << "] End RunImpl → local=" << local_result << ", global=" << global_result
            << ", time=" << elapsed_ms << " ms\n";

  return true;
}

bool SizovDStringMismatchCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace sizov_d_string_mismatch_count
