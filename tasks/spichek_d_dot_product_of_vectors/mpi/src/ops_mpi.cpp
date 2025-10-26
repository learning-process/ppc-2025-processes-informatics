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
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool SpichekDDotProductOfVectorsMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDDotProductOfVectorsMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto input = GetInput();
  if (input == 0) {
    return false;
  }

  // Распределяем работу между процессами
  InType elements_per_process = input / size;
  InType remainder = input % size;
  
  // Определяем диапазон для каждого процесса
  InType start = rank * elements_per_process + (rank < remainder ? rank : remainder);
  InType end = start + elements_per_process + (rank < remainder ? 1 : 0);

  // Создаем локальные части векторов
  std::vector<InType> local_vector1(end - start);
  std::vector<InType> local_vector2(end - start);
  
  // Инициализируем локальные части векторов
  for (InType i = 0; i < local_vector1.size(); ++i) {
    InType global_index = start + i;
    local_vector1[i] = global_index + 1;           // [1, 2, 3, ..., n]
    local_vector2[i] = global_index + 1;           // [1, 2, 3, ..., n]
  }

  // Вычисляем локальное скалярное произведение
  InType local_dot_product = 0;
  for (size_t i = 0; i < local_vector1.size(); ++i) {
    local_dot_product += local_vector1[i] * local_vector2[i];
  }

  // Собираем результаты со всех процессов
  InType global_dot_product = 0;
  MPI_Allreduce(&local_dot_product, &global_dot_product, 1, 
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // Аналог вашего кода: процесс 0 рассылает результат всем процессам
  if (rank == 0) {
    // отправляем всем процессам корректный результат
    for (int i = 1; i < size; i++) {
      MPI_Send(&global_dot_product, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    // сами устанавливаем значение
    GetOutput() = global_dot_product;
  } else {
    // все остальные процессы получают результат
    MPI_Recv(&global_dot_product, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    GetOutput() = global_dot_product;
  }

  // Синхронизация перед завершением
  MPI_Barrier(MPI_COMM_WORLD);

  return GetOutput() > 0;
}

bool SpichekDDotProductOfVectorsMPI::PostProcessingImpl() {
  // Для скалярного произведения постобработка может не требоваться
  return GetOutput() > 0;
}

}  // namespace spichek_d_dot_product_of_vectors
