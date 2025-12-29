#include "goriacheva_k_violation_order_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

//#include <numeric>
#include <vector>

#include "goriacheva_k_violation_order_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

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
  input_vec = GetInput();
  result = 0;
  return true;
}

bool GoriachevaKViolationOrderElemVecMPI::RunImpl() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = static_cast<int>(input_vec.size());

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n <= 1) { 
      result = 0; 
      MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD); 
      return true; 
    }

    int base = n / size, rem = n % size;
    int local_size = base + (rank < rem ? 1 : 0);
    std::vector<int> local(local_size);

    if (rank == 0) {
        std::copy(input_vec.begin(), input_vec.begin() + local_size, local.begin());
        for (int r = 1; r < size; ++r) {
            int sz = base + (r < rem ? 1 : 0);
            if (sz){ 
              MPI_Send(input_vec.data() + r*base + std::min(r, rem), sz, MPI_INT, r, 0, MPI_COMM_WORLD);
            }
        }
    } else if (local_size) {
        MPI_Recv(local.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int local_count = 0;

    for (int i = 0; i + 1 < local_size; ++i){ 
      if (local[i] > local[i+1]){
        ++local_count;
      }
    }

    int left_last = 0, send_val = local_size ? local.back() : 0;

    MPI_Sendrecv(&send_val, 1, MPI_INT, rank+1 < size ? rank+1 : MPI_PROC_NULL, 1,
                 &left_last, 1, MPI_INT, rank > 0 ? rank-1 : MPI_PROC_NULL, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 
    if (local_size && rank > 0 && left_last > local.front()){ 
      ++local_count;
    }

    MPI_Reduce(&local_count, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
}

bool GoriachevaKViolationOrderElemVecMPI::PostProcessingImpl() {
    GetOutput() = result;
    return true;
}

}  // namespace goriacheva_k_violation_order_elem_vec
