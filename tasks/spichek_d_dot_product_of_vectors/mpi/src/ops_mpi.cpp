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

  // Создаем локальные копии векторов
  std::vector<int> local_vec1, local_vec2;

  if (rank == 0) {
    // Root процесс уже имеет данные
    local_vec1 = vector1;
    local_vec2 = vector2;
  } else {
    // Другие процессы выделяют память
    local_vec1.resize(vector_size);
    local_vec2.resize(vector_size);
  }

  // Распространяем данные векторов на все процессы
  MPI_Bcast(local_vec1.data(), vector_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(local_vec2.data(), vector_size, MPI_INT, 0, MPI_COMM_WORLD);

  // Теперь все процессы имеют полные копии векторов
  // Вычисляем свою часть скалярного произведения
  size_t chunk_size = n / size;
  size_t start = rank * chunk_size;
  size_t end = (rank == size - 1) ? n : start + chunk_size;

  int local_dot_product = 0;
  for (size_t i = start; i < end; ++i) {
    local_dot_product += local_vec1[i] * local_vec2[i];
  }

  int global_dot_product = 0;
  MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Только root процесс устанавливает результат
  if (rank == 0) {
    GetOutput() = global_dot_product;
  }

  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
