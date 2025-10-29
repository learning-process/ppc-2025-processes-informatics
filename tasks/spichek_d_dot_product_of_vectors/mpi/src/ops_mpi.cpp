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

  // Локальный размер (0 если данных нет или размеры не совпадают)
  int local_size = 0;
  if (!vector1.empty() && !vector2.empty() && vector1.size() == vector2.size()) {
    local_size = static_cast<int>(vector1.size());
  }

  // Все процессы участвуют в поиске максимального размера
  int max_size = 0;
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  // Если у всех процессов нет данных — корректно возвращаем 0, но не завершаем MPI
  if (max_size == 0) {
    GetOutput() = 0;
    MPI_Barrier(MPI_COMM_WORLD);  // синхронизируем все процессы перед выходом
    return true;
  }

  // Разбиваем работу по всем процессам, основываясь на общем размере max_size
  const size_t n = static_cast<size_t>(max_size);
  const size_t chunk_size = n / static_cast<size_t>(size);
  const size_t start = static_cast<size_t>(rank) * chunk_size;
  const size_t end = (rank == size - 1) ? n : (start + chunk_size);

  // Вычисляем локальный вклад (только если у процесса есть реальные данные)
  long long local_dot = 0;
  if (local_size > 0) {
    // Безопасная проверка границ: если вектор действительно меньше, чем участок,
    // то считаем только до своего размера (но обычно data-owning процесс имеет размер == max_size)
    const size_t local_n = vector1.size();
    const size_t real_start = std::min(start, local_n);
    const size_t real_end = std::min(end, local_n);
    for (size_t i = real_start; i < real_end; ++i) {
      local_dot += static_cast<long long>(vector1[i]) * static_cast<long long>(vector2[i]);
    }
  }

  // Суммируем у всех
  long long global_dot = 0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  // Устанавливаем результат (если OutType — int, возможно нужно изменить тип выхода в интерфейсе)
  GetOutput() = global_dot;

  // Гарантируем, что все процессы завершат вместе (без преждевременного выхода)
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_dot_product_of_vectors
