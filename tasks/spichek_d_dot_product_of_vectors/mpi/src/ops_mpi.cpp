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

  // Определяем локальный размер (если данные корректны)
  int local_size = 0;
  if (!vector1.empty() && !vector2.empty() && vector1.size() == vector2.size()) {
    local_size = static_cast<int>(vector1.size());
  }

  // Находим максимальный размер вектора среди всех процессов
  int max_size = 0;
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  // Если данных нет ни у кого — просто возвращаем 0
  if (max_size == 0) {
    GetOutput() = 0;
    return true;
  }

  // Вычисляем границы участка для данного процесса
  const size_t n = static_cast<size_t>(max_size);
  const size_t chunk_size = n / static_cast<size_t>(size);
  const size_t start = static_cast<size_t>(rank) * chunk_size;
  const size_t end = (rank == size - 1) ? n : start + chunk_size;

  // Локальное вычисление
  long long local_dot = 0;
  if (local_size > 0) {
    const size_t local_n = vector1.size();
    const size_t real_end = std::min(end, local_n);
    for (size_t i = start; i < real_end; ++i) {
      local_dot += static_cast<long long>(vector1[i]) * static_cast<long long>(vector2[i]);
    }
  }

  // Глобальное суммирование
  long long global_dot = 0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  // Устанавливаем результат
  GetOutput() = static_cast<OutType>(global_dot);

  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
