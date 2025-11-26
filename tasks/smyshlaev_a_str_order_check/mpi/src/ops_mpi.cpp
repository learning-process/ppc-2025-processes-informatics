#include "smyshlaev_a_str_order_check/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <string>
#include <vector>

#include "smyshlaev_a_str_order_check/common/include/common.hpp"

namespace smyshlaev_a_str_order_check {

SmyshlaevAStrOrderCheckMPI::SmyshlaevAStrOrderCheckMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }

  GetOutput() = 0;
}

bool SmyshlaevAStrOrderCheckMPI::ValidationImpl() {
  return true;
}

bool SmyshlaevAStrOrderCheckMPI::PreProcessingImpl() {
  return true;
}

bool SmyshlaevAStrOrderCheckMPI::RunImpl() {
  int proc_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int len1 = 0;
  int len2 = 0;

  if (rank == 0) {
    const auto &input_data = GetInput();
    len1 = static_cast<int>(input_data.first.length());
    len2 = static_cast<int>(input_data.second.length());
  }

  MPI_Bcast(&len1, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&len2, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int min_len = static_cast<int>(std::min(len1, len2));

  if (proc_count > min_len) {
    if (rank == 0) {
      const auto &str1 = GetInput().first;
      const auto &str2 = GetInput().second;
      int res = 0;
      for (int i = 0; i < min_len; ++i) {
        if (str1[i] < str2[i]) {
          res = -1;
          break;
        }
        if (str1[i] > str2[i]) {
          res = 1;
          break;
        }
      }
      if (res == 0) {
        if (len1 < len2) {
          res = -1;
        } else if (len1 > len2) {
          res = 1;
        }
      }
      GetOutput() = res;
    } else {
      GetOutput() = 0;
    }

    int output = GetOutput();
    MPI_Bcast(&output, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = output;
    return true;
  }

  std::vector<int> sendcounts(proc_count);
  std::vector<int> offsets(proc_count);

  const int part = min_len / proc_count;
  const int remainder = min_len % proc_count;

  int offset = 0;
  for (int i = 0; i < proc_count; i++) {
    sendcounts[i] = part + (i < remainder ? 1 : 0);
    offsets[i] = offset;
    offset += sendcounts[i];
  }

  int local_size = sendcounts[rank];
  std::vector<char> local_str1(local_size);
  std::vector<char> local_str2(local_size);
  MPI_Scatterv(rank == 0 ? GetInput().first.data() : nullptr, sendcounts.data(), offsets.data(), MPI_CHAR,
               local_str1.data(), local_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? GetInput().second.data() : nullptr, sendcounts.data(), offsets.data(), MPI_CHAR,
               local_str2.data(), local_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_result = 0;
  for (int i = 0; i < local_size; ++i) {
    if (local_str1[i] < local_str2[i]) {
      local_result = -1;
      break;
    }
    if (local_str1[i] > local_str2[i]) {
      local_result = 1;
      break;
    }
  }

  std::vector<int> all_results(proc_count);
  MPI_Allgather(&local_result, 1, MPI_INT, all_results.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int global_result = 0;
  for (int res : all_results) {
    if (res != 0) {
      global_result = res;
      break;
    }
  }

  if (global_result == 0) {
    if (len1 < len2) {
      global_result = -1;
    } else if (len1 > len2) {
      global_result = 1;
    }
  }

  GetOutput() = global_result;

  return true;
}

bool SmyshlaevAStrOrderCheckMPI::PostProcessingImpl() {
  return true;
}

}  // namespace smyshlaev_a_str_order_check
