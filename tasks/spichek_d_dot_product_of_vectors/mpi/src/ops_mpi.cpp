#include "spichek_d_dot_product_of_vectors/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>

#include "spichek_d_dot_product_of_vectors/common/include/common.hpp"

namespace spichek_d_dot_product_of_vectors {

SpichekDDotProductOfVectorsMPI::SpichekDDotProductOfVectorsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDDotProductOfVectorsMPI::ValidationImpl() {
  const auto &[vector1, vector2] = GetInput();
  // Разрешаем пустые векторы, чтобы все процессы участвовали в MPI-вызовах.
  return (vector1.size() == vector2.size());
}

bool SpichekDDotProductOfVectorsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[vector1, vector2] = GetInput();

  // Локальный размер (0 если данных нет или размеры не совпадают)
  int local_size =
      (!vector1.empty() && !vector2.empty() && vector1.size() == vector2.size()) ? static_cast<int>(vector1.size()) : 0;

  // Узнаём максимальный размер у всех процессов
  int max_size = 0;
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  // Если у всех нет данных — результат 0
  if (max_size == 0) {
    GetOutput() = 0;
    return true;
  }

  // Корректное разделение с учётом остатка
  const auto n = static_cast<size_t>(max_size);
  const size_t base_chunk = n / static_cast<size_t>(size);
  const size_t remainder = n % static_cast<size_t>(size);
  const size_t start = (static_cast<size_t>(rank) * base_chunk) + std::min(static_cast<size_t>(rank), remainder);
  const size_t end = start + base_chunk + (static_cast<size_t>(rank) < static_cast<size_t>(remainder) ? 1u : 0u);

  long long local_dot = 0;
  if (local_size > 0) {
    const size_t local_n = vector1.size();
    const size_t real_start = std::min(start, local_n);
    const size_t real_end = std::min(end, local_n);
    for (size_t i = real_start; i < real_end; ++i) {
      local_dot += static_cast<long long>(vector1[i]) * static_cast<long long>(vector2[i]);
    }
  }

  long long global_dot = 0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_dot);

  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
