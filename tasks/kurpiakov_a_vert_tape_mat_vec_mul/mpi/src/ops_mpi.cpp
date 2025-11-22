#include "kurpiakov_a_vert_tape_mat_vec_mul/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <utility>
#include <vector>

#include "kurpiakov_a_vert_tape_mat_vec_mul/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kurpiakov_a_vert_tape_mat_vec_mul {

KurpiakovAVretTapeMulMPI::KurpiakovAVretTapeMulMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KurpiakovAVretTapeMulMPI::ValidationImpl() {
  // int size_of_vec = std::get<2>(GetInput()).size();
  // int size_of_mat = std::get<1>(GetInput()).size();
  int size_of_input = std::get<0>(GetInput());
  // bool res = std::cmp_equal(size_of_vec / size_of_input, size_of_mat / (size_of_input * size_of_input));
  return (size_of_input >= 0);
}

bool KurpiakovAVretTapeMulMPI::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool KurpiakovAVretTapeMulMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = std::get<0>(GetInput());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    GetOutput() = OutType{};
    return true;
  }

  const int base_cols = total_size / size;
  const int remainder = total_size % size;
  const int local_cols = base_cols + (rank < remainder);
  const int col_displ = rank * base_cols + (rank < remainder ? rank : remainder);

  OutType local_vec(total_size);
  if (rank == 0) {
    local_vec = std::get<2>(GetInput());
  }
  MPI_Bcast(local_vec.data(), total_size, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  OutType local_matrix;

  if (rank == 0) {
    const auto &global_mat = std::get<1>(GetInput());

    std::vector<MPI_Request> requests;
    for (int dest = 1; dest < size; ++dest) {
      int dest_cols = base_cols + (dest < remainder);
      if (dest_cols > 0) {
        int dest_displ = dest * base_cols + (dest < remainder ? dest : remainder);
        requests.emplace_back();
        MPI_Isend(global_mat.data() + dest_displ * total_size, dest_cols * total_size, MPI_LONG_LONG, dest, 0,
                  MPI_COMM_WORLD, &requests.back());
      }
    }

    if (local_cols > 0) {
      local_matrix.resize(local_cols * total_size);
      memcpy(local_matrix.data(), global_mat.data() + col_displ * total_size,
             local_cols * total_size * sizeof(long long));
    }

    if (!requests.empty()) {
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

  } else if (local_cols > 0) {
    local_matrix.resize(local_cols * total_size);
    MPI_Recv(local_matrix.data(), local_cols * total_size, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  OutType local_result(total_size, 0);
  long long sum;
  if (local_cols > 0) {
    for (int i = 0; i < total_size; ++i) {
      sum = 0;
      for (int j = 0; j < local_cols; ++j) {
        sum += local_matrix[i + j * total_size] * local_vec[col_displ + j];
      }
      local_result[i] = sum;
    }
  }

  OutType res_vec(total_size);

  MPI_Allreduce(local_result.data(), res_vec.data(), total_size, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = res_vec;
  return true;
}

bool KurpiakovAVretTapeMulMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kurpiakov_a_vert_tape_mat_vec_mul
