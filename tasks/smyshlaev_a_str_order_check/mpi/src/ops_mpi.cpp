#include "smyshlaev_a_str_order_check/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <string>

#include "smyshlaev_a_str_order_check/common/include/common.hpp"

namespace smyshlaev_a_str_order_check {

SmyshlaevAStrOrderCheckMPI::SmyshlaevAStrOrderCheckMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SmyshlaevAStrOrderCheckMPI::ValidationImpl() {
  return true;
}

bool SmyshlaevAStrOrderCheckMPI::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAStrOrderCheckMPI::RunImpl() {
  const auto &input_data = GetInput();
  const std::string &str1 = input_data.first;
  const std::string &str2 = input_data.second;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int min_len = std::min(str1.length(), str2.length());
  const int chunk_size = (size > 0) ? (min_len / size) : 0;
  const int remainder = (size > 0) ? (min_len % size) : 0;

  const int start_idx = rank * chunk_size + std::min(rank, remainder);
  const int end_idx = start_idx + chunk_size + (rank < remainder ? 1 : 0);

  int local_result = 0;
  for (int i = start_idx; i < end_idx; ++i) {
    if (str1[i] < str2[i]) {
      local_result = -1;
      break;
    }
    if (str1[i] > str2[i]) {
      local_result = 1;
      break;
    }
  }

  if (rank == 0) {
    std::vector<int> all_results(size);
    MPI_Gather(&local_result, 1, MPI_INT, all_results.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int global_result = 0;
    for (int result : all_results) {
      if (result != 0) {
        global_result = result;
        break;
      }
    }

    if (global_result == 0) {
      if (str1.length() < str2.length()) {
        global_result = -1;
      } else if (str1.length() > str2.length()) {
        global_result = 1;
      }
    }
    GetOutput() = global_result;

  } else {
    MPI_Gather(&local_result, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool SmyshlaevAStrOrderCheckMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_str_order_check
