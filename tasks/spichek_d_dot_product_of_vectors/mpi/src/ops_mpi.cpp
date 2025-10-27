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
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[vector1, vector2] = GetInput();

  if (vector1.size() != vector2.size() || vector1.empty()) {
    return false;
  }

  size_t n = vector1.size();

  // Распространяем размер векторов на все процессы
  size_t vector_size = n;
  MPI_Bcast(&vector_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  // Все процессы создают локальные копии одинакового размера
  std::vector<int> local_vec1(vector_size);
  std::vector<int> local_vec2(vector_size);

  if (rank == 0) {
    local_vec1 = vector1;
    local_vec2 = vector2;
  }

  // Распространяем данные векторов на все процессы
  MPI_Bcast(local_vec1.data(), vector_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(local_vec2.data(), vector_size, MPI_INT, 0, MPI_COMM_WORLD);

  // Простое распределение (как в исходном коде)
  size_t chunk_size = vector_size / size;
  size_t start = rank * chunk_size;
  size_t end = (rank == size - 1) ? vector_size : start + chunk_size;

  int local_dot_product = 0;
  for (size_t i = start; i < end; ++i) {
    local_dot_product += local_vec1[i] * local_vec2[i];
  }

  int global_dot_product = 0;
  MPI_Allreduce(&local_dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_dot_product;

  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
