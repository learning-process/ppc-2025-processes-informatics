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

  // Определяем, кто имеет данные
  int data_size = 0;
  if (!vector1.empty() && !vector2.empty() && vector1.size() == vector2.size()) {
    data_size = static_cast<int>(vector1.size());
  }

  // Находим максимальный размер данных среди всех процессов
  int max_data_size;
  MPI_Allreduce(&data_size, &max_data_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (max_data_size == 0) {
    GetOutput() = 0;
    return false;
  }

  // Если у процесса нет данных, он вычисляет 0
  int local_dot_product = 0;
  if (data_size > 0) {
    size_t n = vector1.size();
    size_t chunk_size = n / size;
    size_t start = rank * chunk_size;
    size_t end = (rank == size - 1) ? n : start + chunk_size;

    for (size_t i = start; i < end; ++i) {
      local_dot_product += vector1[i] * vector2[i];
    }
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
