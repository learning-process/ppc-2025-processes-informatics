#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"
#include "util/include/util.hpp"

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsMPI::SpichekDDotProductOfVectorsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsMPI::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  return (!vector1.empty()) && (vector1.size() == vector2.size()) && (GetOutput() == 0);
}

bool SpichekDDotProductOfVectorsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[vector1, vector2] = GetInput();

  int local_size =
      (!vector1.empty() && !vector2.empty() && vector1.size() == vector2.size()) ? static_cast<int>(vector1.size()) : 0;

  int max_size = 0;
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  long long local_dot = 0;
  if (max_size > 0 && local_size > 0) {
    size_t n = vector1.size();
    size_t chunk_size = n / size;
    size_t start = rank * chunk_size;
    size_t end = (rank == size - 1) ? n : start + chunk_size;

    for (size_t i = start; i < end; ++i) {
      local_dot += static_cast<long long>(vector1[i]) * static_cast<long long>(vector2[i]);
    }
  }

  long long global_dot = 0;

  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  if (max_size == 0) {
    GetOutput() = 0;
  } else {
    GetOutput() = static_cast<OutType>(global_dot);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
