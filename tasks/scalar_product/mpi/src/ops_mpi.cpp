#include "scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "scalar_product/common/include/common.hpp"
#include "util/include/util.hpp"

namespace scalar_product {

ScalarProductMPI::ScalarProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ScalarProductMPI::ValidationImpl() {
  const auto &input_pair = GetInput();
  const auto &vector_a = input_pair.first;
  const auto &vector_b = input_pair.second;
  return vector_a.size() == vector_b.size() && !vector_a.empty();
}

bool ScalarProductMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ScalarProductMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &input_pair = GetInput();
  const auto &vector_a = input_pair.first;
  const auto &vector_b = input_pair.second;
  
  if (vector_a.empty() || vector_b.empty()) {
    return false;
  }
  
  size_t vector_size = vector_a.size();

  size_t chunk_size = vector_size / world_size;
  size_t remainder = vector_size % world_size;

  size_t start_index = rank * chunk_size + std::min<size_t>(rank, remainder);
  size_t end_index = start_index + chunk_size + (rank < remainder ? 1 : 0);

  int local_scalar_product = 0;
  for (size_t i = start_index; i < end_index; ++i) {
    local_scalar_product += vector_a[i] * vector_b[i];
  }

  int global_scalar_product = 0;
  MPI_Reduce(&local_scalar_product, &global_scalar_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_scalar_product;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ScalarProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace scalar_product