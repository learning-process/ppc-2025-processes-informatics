#include "goriacheva_k_violation_order_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"

namespace goriacheva_k_violation_order_elem_vec {

GoriachevaKViolationOrderElemVecMPI::GoriachevaKViolationOrderElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GoriachevaKViolationOrderElemVecMPI::ValidationImpl() {
  return true;
}

bool GoriachevaKViolationOrderElemVecMPI::PreProcessingImpl() {
  input_vec_ = GetInput();
  result_ = 0;
  return true;
}

bool GoriachevaKViolationOrderElemVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = static_cast<int>(input_vec_.size());
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    result_ = 0;
    MPI_Bcast(&result_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  int base = n / size;
  int rem = n % size;
  int local_size = base + (rank < rem ? 1 : 0);
  std::vector<int> local(local_size);

  if (rank == 0) {
    std::copy(input_vec_.begin(), input_vec_.begin() + local_size, local.begin());
    for (int rank_iter = 1; rank_iter < size; ++rank_iter) {
      int sz = base + (rank_iter < rem ? 1 : 0);
      if (sz != 0) {
        MPI_Send(input_vec_.data() + static_cast<ptrdiff_t>((rank_iter * base) + std::min(rank_iter, rem)), sz, MPI_INT,
                 rank_iter, 0, MPI_COMM_WORLD);
      }
    }
  } else if (local_size > 0) {
    MPI_Recv(local.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int local_count = 0;
  for (int i = 0; i + 1 < local_size; ++i) {
    if (local[i] > local[i + 1]) {
      ++local_count;
    }
  }

  int send_val = (local_size > 0) ? local.back() : 0;
  int left_last = 0;

  MPI_Sendrecv(&send_val, 1, MPI_INT, (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL, 1, &left_last, 1, MPI_INT,
               (rank > 0) ? rank - 1 : MPI_PROC_NULL, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (rank > 0 && local_size > 0 && left_last > local.front()) {
    ++local_count;
  }

  MPI_Reduce(&local_count, &result_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&result_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool GoriachevaKViolationOrderElemVecMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace goriacheva_k_violation_order_elem_vec
