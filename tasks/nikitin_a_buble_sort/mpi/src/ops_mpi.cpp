#include "nikitin_a_buble_sort/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "nikitin_a_buble_sort/common/include/common.hpp"

namespace nikitin_a_buble_sort {

namespace {

// Вычисление распределения данных между процессами
void CalculateDistribution(int total_elements, int num_processes, 
                          std::vector<int>& counts, std::vector<int>& displacements) {
  int base_count = total_elements / num_processes;
  int remainder = total_elements % num_processes;
  
  int offset = 0;
  for (int i = 0; i < num_processes; ++i) {
    counts[i] = base_count + (i < remainder ? 1 : 0);
    displacements[i] = offset;
    offset += counts[i];
  }
}

// Локальная сортировка для четных или нечетных индексов
void LocalOddEvenSort(std::vector<double>& local_data, int global_start_index, int parity) {
  int local_size = static_cast<int>(local_data.size());
  
  for (int i = 0; i < local_size - 1; ++i) {
    int global_index = global_start_index + i;
    
    // Сортируем только элементы с заданной четностью
    if ((global_index % 2) == parity) {
      if (local_data[i] > local_data[i + 1]) {
        std::swap(local_data[i], local_data[i + 1]);
      }
    }
  }
}

// Обмен граничными элементами между соседними процессами
void ExchangeBoundaryValues(std::vector<double>& local_data, const std::vector<int>& counts, 
                           int current_rank, int neighbor_rank) {
  int local_size = static_cast<int>(local_data.size());
  int neighbor_size = counts[neighbor_rank];
  
  if (local_size == 0 || neighbor_size == 0) {
    return;
  }
  
  bool is_left_process = (current_rank < neighbor_rank);
  double value_to_send = is_left_process ? local_data[local_size - 1] : local_data[0];
  double received_value = 0.0;
  
  // Обмен граничными значениями
  MPI_Sendrecv(&value_to_send, 1, MPI_DOUBLE, neighbor_rank, 0,
               &received_value, 1, MPI_DOUBLE, neighbor_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  // Корректировка граничных значений
  if (is_left_process) {
    // Левый процесс оставляет минимальное из двух значений
    if (local_data[local_size - 1] > received_value) {
      local_data[local_size - 1] = received_value;
    }
  } else {
    // Правый процесс оставляет максимальное из двух значений
    if (local_data[0] < received_value) {
      local_data[0] = received_value;
    }
  }
}

// Одна фаза четно-нечетной перестановки
void PerformOddEvenPhase(std::vector<double>& local_data, 
                        const std::vector<int>& counts,
                        const std::vector<int>& displacements,
                        int current_rank,
                        int total_processes,
                        int phase_number) {
  if (local_data.empty()) {
    return;
  }
  
  // Определяем четность фазы (0 - четная, 1 - нечетная)
  int parity = phase_number % 2;
  int global_start = displacements[current_rank];
  
  // Выполняем локальную сортировку
  LocalOddEvenSort(local_data, global_start, parity);
  
  // Определяем соседа для обмена
  int neighbor = -1;
  bool is_even_phase = (phase_number % 2 == 0);
  bool is_even_rank = (current_rank % 2 == 0);
  
  if (is_even_phase == is_even_rank) {
    // Для четной фазы и четного ранга (или нечетной и нечетного) обмен с правым соседом
    neighbor = current_rank + 1;
  } else {
    // Для четной фазы и нечетного ранга (или нечетной и четного) обмен с левым соседом
    neighbor = current_rank - 1;
  }
  
  // Выполняем обмен с соседом, если он существует
  if (neighbor >= 0 && neighbor < total_processes) {
    ExchangeBoundaryValues(local_data, counts, current_rank, neighbor);
  }
}

}  // namespace

NikitinABubleSortMPI::NikitinABubleSortMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool NikitinABubleSortMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Только процесс 0 проверяет входные данные
  if (rank == 0) {
    return true;  // Разрешаем любой массив, включая пустой
  }
  return true;
}

bool NikitinABubleSortMPI::PreProcessingImpl() {
  // Копируем входные данные
  GetOutput() = GetInput();
  return true;
}

bool NikitinABubleSortMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Получаем размер массива
  int total_elements = 0;
  if (rank == 0) {
    total_elements = static_cast<int>(GetOutput().size());
  }
  
  // Рассылаем размер массива всем процессам
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  // Обработка особых случаев
  if (total_elements <= 1) {
    if (rank == 0) {
      // Ничего не делаем - массив уже отсортирован
      return true;
    } else {
      GetOutput().clear();
      return true;
    }
  }
  
  // Вычисляем распределение данных
  std::vector<int> counts(size);
  std::vector<int> displacements(size);
  CalculateDistribution(total_elements, size, counts, displacements);
  
  // Получаем локальный размер
  int local_size = counts[rank];
  std::vector<double> local_data(local_size);
  
  // Разделяем данные между процессами
  double* global_data_ptr = (rank == 0) ? GetOutput().data() : nullptr;
  MPI_Scatterv(global_data_ptr, counts.data(), displacements.data(), MPI_DOUBLE,
               local_data.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  // Выполняем четно-нечетную сортировку
  // Максимальное количество фаз = количеству элементов
  for (int phase = 0; phase < total_elements; ++phase) {
    PerformOddEvenPhase(local_data, counts, displacements, rank, size, phase);
  }
  
  // Собираем результаты на процессе 0
  std::vector<double> sorted_result;
  if (rank == 0) {
    sorted_result.resize(total_elements);
  }
  
  MPI_Gatherv(local_data.data(), local_size, MPI_DOUBLE,
              sorted_result.data(), counts.data(), displacements.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
  
  // Рассылаем отсортированный результат всем процессам
  if (rank == 0) {
    GetOutput() = sorted_result;
  } else {
    GetOutput().resize(total_elements);
  }
  
  MPI_Bcast(GetOutput().data(), total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  return true;
}

bool NikitinABubleSortMPI::PostProcessingImpl() {
  // Проверяем корректность сортировки
  const std::vector<double>& result = GetOutput();
  
  if (result.empty()) {
    return true;
  }
  
  for (size_t i = 1; i < result.size(); ++i) {
    if (result[i - 1] > result[i]) {
      return false;
    }
  }
  
  return true;
}

}  // namespace nikitin_a_buble_sort
