#include "egashin_k_lexicographical_check/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "egashin_k_lexicographical_check/common/include/common.hpp"

namespace egashin_k_lexicographical_check {

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = false;
}

bool TestTaskMPI::ValidationImpl() {
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  return true;
}

bool TestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int min_len = 0;
  int s1_len = 0;
  int s2_len = 0;

  if (rank == 0) {
    s1_len = static_cast<int>(GetInput().first.size());
    s2_len = static_cast<int>(GetInput().second.size());
    min_len = (s1_len < s2_len) ? s1_len : s2_len;
  }

  if (MPI_Bcast(&min_len, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
    return false;
  }

  if (min_len == 0) {
    if (rank == 0) {
      GetOutput() = (s1_len < s2_len);
    }
    return true;
  }

  int delta = min_len / size;
  int remainder = min_len % size;

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      counts[i] = delta + (i < remainder ? 1 : 0);
      displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }
  }

  int local_count = 0;
  MPI_Scatter(counts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<char> local_s1(local_count);
  std::vector<char> local_s2(local_count);

  const char *send_s1 = nullptr;
  const char *send_s2 = nullptr;

  if (rank == 0) {
    send_s1 = GetInput().first.data();
    send_s2 = GetInput().second.data();
  }

  MPI_Scatterv(send_s1, counts.data(), displs.data(), MPI_CHAR, local_s1.data(), local_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  MPI_Scatterv(send_s2, counts.data(), displs.data(), MPI_CHAR, local_s2.data(), local_count, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  int local_res = 0;
  if (local_count > 0) {
    for (int i = 0; i < local_count; ++i) {
      auto c1 = static_cast<unsigned char>(local_s1[i]);
      auto c2 = static_cast<unsigned char>(local_s2[i]);
      if (c1 < c2) {
        local_res = -1;
        break;
      }
      if (c1 > c2) {
        local_res = 1;
        break;
      }
    }
  }

  std::vector<int> global_results(size);
  MPI_Gather(&local_res, 1, MPI_INT, global_results.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int final_decision = 0;
    for (int i = 0; i < size; ++i) {
      if (global_results[i] != 0) {
        final_decision = global_results[i];
        break;
      }
    }

    if (final_decision != 0) {
      GetOutput() = (final_decision == -1);
    } else {
      GetOutput() = (s1_len < s2_len);
    }
  }

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egashin_k_lexicographical_check
