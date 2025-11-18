#include "leonova_a_most_diff_neigh_vec_elems/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

namespace leonova_a_most_diff_neigh_vec_elems {

LeonovaAMostDiffNeighVecElemsMPI::LeonovaAMostDiffNeighVecElemsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::tuple<int, int>(0, 0);
}

bool LeonovaAMostDiffNeighVecElemsMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool LeonovaAMostDiffNeighVecElemsMPI::PreProcessingImpl() {
  return true;
}

bool LeonovaAMostDiffNeighVecElemsMPI::RunImpl() {
  const auto &input_vec = GetInput();

  if (!ValidationImpl()) {
    return false;
  }

  if (input_vec.size() == 1) {
    GetOutput() = std::make_tuple(input_vec[0], input_vec[0]);
    return true;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = static_cast<int>(input_vec.size());
  if (size > 1) {
    ProcessWithMultipleProcesses(rank, size, total_size, input_vec);
  } else {
    ProcessSequentially(input_vec);
  }

  return true;
}

void LeonovaAMostDiffNeighVecElemsMPI::ProcessWithMultipleProcesses(int rank, int size, int total_size,
                                                                    const std::vector<int> &input_vec) {
  int actual_processes = std::min(size, total_size);
  int local_max_diff = -1;
  int local_first = 0;
  int local_second = 0;

  if (rank < actual_processes) {
    ProcessLocalData(rank, actual_processes, total_size, input_vec, local_max_diff, local_first, local_second);
  }

  GatherAndProcessResults(rank, actual_processes, local_max_diff, local_first, local_second, size);
}

void LeonovaAMostDiffNeighVecElemsMPI::ProcessLocalData(int rank, int actual_processes, int total_size,
                                                        const std::vector<int> &input_vec, int &local_max_diff,
                                                        int &local_first, int &local_second) {
  int chunk_size = total_size / actual_processes;
  int remainder = total_size % actual_processes;

  int my_size = chunk_size + (rank < remainder ? 1 : 0) + 1;
  int my_offset = (rank * chunk_size) + std::min(rank, remainder);

  if (rank == actual_processes - 1) {
    my_size = total_size - my_offset;
  }

  std::vector<int> local_data(my_size);
  ReceiveLocalData(rank, actual_processes, input_vec, my_size, local_data, total_size);

  FindLocalMaxDiff(local_data, local_max_diff, local_first, local_second);
}

void LeonovaAMostDiffNeighVecElemsMPI::ReceiveLocalData(int rank, int actual_processes,
                                                        const std::vector<int> &input_vec, int my_size,
                                                        std::vector<int> &local_data, int total_size) {
  if (rank == 0) {
    for (int index = 0; index < my_size; ++index) {
      local_data[index] = input_vec[index];
    }

    for (int dest = 1; dest < actual_processes; ++dest) {
      SendDataToProcess(dest, actual_processes, input_vec, total_size);
    }
  } else {
    MPI_Recv(local_data.data(), my_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void LeonovaAMostDiffNeighVecElemsMPI::SendDataToProcess(int dest, int actual_processes,
                                                         const std::vector<int> &input_vec, int total_size) {
  int chunk_size = total_size / actual_processes;
  int remainder = total_size % actual_processes;

  int dest_size = chunk_size + (dest < remainder ? 1 : 0) + 1;
  int dest_offset = (dest * chunk_size) + std::min(dest, remainder);

  if (dest == actual_processes - 1) {
    dest_size = total_size - dest_offset;
  }

  MPI_Send(input_vec.data() + dest_offset, dest_size, MPI_INT, dest, 0, MPI_COMM_WORLD);
}

void LeonovaAMostDiffNeighVecElemsMPI::FindLocalMaxDiff(const std::vector<int> &local_data, int &local_max_diff,
                                                        int &local_first, int &local_second) {
  for (int index = 0; index < static_cast<int>(local_data.size()) - 1; ++index) {
    int diff = std::abs(local_data[index] - local_data[index + 1]);
    if (diff > local_max_diff) {
      local_max_diff = diff;
      local_first = local_data[index];
      local_second = local_data[index + 1];
    }
  }
}

void LeonovaAMostDiffNeighVecElemsMPI::GatherAndProcessResults(int rank, int actual_processes, int local_max_diff,
                                                               int local_first, int local_second, int size) {
  if (rank >= actual_processes) {
    local_max_diff = -1;
    local_first = 0;
    local_second = 0;
  }

  std::vector<int> all_diffs(size);
  std::vector<int> all_firsts(size);
  std::vector<int> all_seconds(size);

  MPI_Gather(&local_max_diff, 1, MPI_INT, all_diffs.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_first, 1, MPI_INT, all_firsts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_second, 1, MPI_INT, all_seconds.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    FindGlobalMaxDiff(all_diffs, all_firsts, all_seconds, actual_processes);
  }

  BroadcastResult(rank);
}

void LeonovaAMostDiffNeighVecElemsMPI::FindGlobalMaxDiff(const std::vector<int> &all_diffs,
                                                         const std::vector<int> &all_firsts,
                                                         const std::vector<int> &all_seconds, int actual_processes) {
  int global_max_diff = -1;
  int global_first = 0;
  int global_second = 0;

  for (int index = 0; index < actual_processes; ++index) {
    if (all_diffs[index] > global_max_diff) {
      global_max_diff = all_diffs[index];
      global_first = all_firsts[index];
      global_second = all_seconds[index];
    }
  }

  GetOutput() = std::make_tuple(global_first, global_second);
}

void LeonovaAMostDiffNeighVecElemsMPI::BroadcastResult(int rank) {
  int result_first = std::get<0>(GetOutput());
  int result_second = std::get<1>(GetOutput());

  MPI_Bcast(&result_first, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&result_second, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput() = std::make_tuple(result_first, result_second);
  }
}

void LeonovaAMostDiffNeighVecElemsMPI::ProcessSequentially(const std::vector<int> &input_vec) {
  int max_diff = -1;
  std::tuple<int, int> best_pair(0, 0);

  for (std::size_t index = 0; index < input_vec.size() - 1; ++index) {
    int diff = std::abs(input_vec[index] - input_vec[index + 1]);
    if (diff > max_diff) {
      max_diff = diff;
      best_pair = std::make_tuple(input_vec[index], input_vec[index + 1]);
    }
  }

  GetOutput() = best_pair;
}

bool LeonovaAMostDiffNeighVecElemsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace leonova_a_most_diff_neigh_vec_elems
