#include "shkrebko_m_shell_sort_batcher_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "shkrebko_m_shell_sort_batcher_merge/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkrebko_m_shell_sort_batcher_merge {

ShkrebkoMShellSortBatcherMergeMPI::ShkrebkoMShellSortBatcherMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int world_rank = 0;
  int err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    GetInput() = in;
  }
}

bool ShkrebkoMShellSortBatcherMergeMPI::ValidationImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  if (world_rank == 0) {
    if (GetInput().empty()) return false;
    // Для упрощения требуем, чтобы размер был кратен количеству процессов
    if (GetInput().size() % world_size != 0) return false;
  }
  
  return true;
}

bool ShkrebkoMShellSortBatcherMergeMPI::PreProcessingImpl() {
  return true;
}

void ShkrebkoMShellSortBatcherMergeMPI::ShellSort(std::vector<int> &arr) {
  size_t n = arr.size();
  if (n <= 1) return;

  std::vector<int> gaps;
  int k = 0;
  int gap_val;
  do {
    if (k % 2 == 0) {
      gap_val = static_cast<int>(9 * std::pow(2, k) - 9 * std::pow(2, k / 2) + 1);
    } else {
      gap_val = static_cast<int>(8 * std::pow(2, k) - 6 * std::pow(2, (k + 1) / 2) + 1);
    }
    if (gap_val > 0 && static_cast<size_t>(gap_val) < n) {
      gaps.push_back(gap_val);
    }
    k++;
  } while (gap_val > 0 && static_cast<size_t>(gap_val) < n);

  std::sort(gaps.rbegin(), gaps.rend());
  
  for (int gap_value : gaps) {
    for (size_t i = gap_value; i < n; ++i) {
      int temp = arr[i];
      size_t j = i;
      while (j >= static_cast<size_t>(gap_value) && arr[j - gap_value] > temp) {
        arr[j] = arr[j - gap_value];
        j -= gap_value;
      }
      arr[j] = temp;
    }
  }
}

int ShkrebkoMShellSortBatcherMergeMPI::CalculateLocalSize(int total_size, int rank, int world_size) {
  int base_size = total_size / world_size;
  if (rank < total_size % world_size) {
    return base_size + 1;
  }
  return base_size;
}

std::vector<int> ShkrebkoMShellSortBatcherMergeMPI::CalculateInterval(int total_size, int rank, int world_size) {
  std::vector<int> interval(2);
  int base_size = total_size / world_size;
  int remainder = total_size % world_size;
  
  int start = rank * base_size;
  if (rank < remainder) {
    start += rank;
  } else {
    start += remainder;
  }
  
  int end = start + base_size - 1;
  if (rank < remainder) {
    end += 1;
  }
  
  interval[0] = start;
  interval[1] = end;
  return interval;
}

void ShkrebkoMShellSortBatcherMergeMPI::MergeAndSplit(std::vector<int> &a, std::vector<int> &b, bool keep_smaller) {
  size_t a_size = a.size();
  size_t b_size = b.size();
  std::vector<int> merged(a_size + b_size);
  
  std::merge(a.begin(), a.end(), b.begin(), b.end(), merged.begin());
  
  if (keep_smaller) {
    // Оставляем меньшие элементы в a
    std::copy(merged.begin(), merged.begin() + static_cast<int64_t>(a_size), a.begin());
    // Большие элементы идут в b
    std::copy(merged.begin() + static_cast<int64_t>(a_size), merged.end(), b.begin());
  } else {
    // Оставляем большие элементы в a
    std::copy(merged.end() - static_cast<int64_t>(a_size), merged.end(), a.begin());
    // Меньшие элементы идут в b
    std::copy(merged.begin(), merged.begin() + static_cast<int64_t>(b_size), b.begin());
  }
}

void ShkrebkoMShellSortBatcherMergeMPI::ExchangeData(std::vector<int> &local_data, int rank, int neighbor_rank, 
                           int total_size, int world_size) {
  MPI_Status status;
  
  // Получаем размер данных соседа
  int neighbor_size = CalculateLocalSize(total_size, neighbor_rank, world_size);
  std::vector<int> neighbor_data(neighbor_size);
  
  // Обмен данными
  MPI_Sendrecv(local_data.data(), static_cast<int>(local_data.size()), MPI_INT,
               neighbor_rank, 0, neighbor_data.data(), neighbor_size, MPI_INT,
               neighbor_rank, 0, MPI_COMM_WORLD, &status);
  
  // Определяем, какие элементы оставить
  // Процесс с меньшим рангом получает меньшие элементы
  bool keep_smaller = (rank < neighbor_rank);
  MergeAndSplit(local_data, neighbor_data, keep_smaller);
}

void ShkrebkoMShellSortBatcherMergeMPI::EvenPhase(std::vector<int> &local_data, int rank, int world_size, int total_size) {
  if (rank % 2 == 0) {
    // Четный процесс обменивается с правым соседом
    if (rank + 1 < world_size) {
      ExchangeData(local_data, rank, rank + 1, total_size, world_size);
    }
  } else {
    // Нечетный процесс обменивается с левым соседом
    if (rank - 1 >= 0) {
      ExchangeData(local_data, rank, rank - 1, total_size, world_size);
    }
  }
}

void ShkrebkoMShellSortBatcherMergeMPI::OddPhase(std::vector<int> &local_data, int rank, int world_size, int total_size) {
  if (rank % 2 == 1) {
    // Нечетный процесс обменивается с правым соседом
    if (rank + 1 < world_size) {
      ExchangeData(local_data, rank, rank + 1, total_size, world_size);
    }
  } else if (rank > 0) {
    // Четный процесс (кроме 0) обменивается с левым соседом
    ExchangeData(local_data, rank, rank - 1, total_size, world_size);
  }
}

void ShkrebkoMShellSortBatcherMergeMPI::BatcherMerge(std::vector<int> &local_data, int rank, int world_size, int total_size) {
  // Количество фаз: достаточно выполнить world_size фаз
  for (int phase = 0; phase < world_size; ++phase) {
    if (phase % 2 == 0) {
      EvenPhase(local_data, rank, world_size, total_size);
    } else {
      OddPhase(local_data, rank, world_size, total_size);
    }
  }
}

bool ShkrebkoMShellSortBatcherMergeMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  // 1. Передача размера массива всем процессам
  int total_size = 0;
  if (world_rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  // 2. Передача всего массива всем процессам
  std::vector<int> full_data;
  if (world_rank == 0) {
    full_data = GetInput();
  } else {
    full_data.resize(total_size);
  }
  MPI_Bcast(full_data.data(), total_size, MPI_INT, 0, MPI_COMM_WORLD);
  
  // 3. Вычисление интервала для текущего процесса
  std::vector<int> interval = CalculateInterval(total_size, world_rank, world_size);
  std::vector<int> local_data;
  
  // 4. Извлечение локальных данных
  if (interval[0] <= interval[1]) {
    local_data = std::vector<int>(full_data.begin() + interval[0], 
                                  full_data.begin() + interval[1] + 1);
  }
  
  // 5. Локальная сортировка Шелла
  if (!local_data.empty()) {
    ShellSort(local_data);
  }
  
  // 6. Четно-нечетное слияние Бэтчера
  BatcherMerge(local_data, world_rank, world_size, total_size);
  
  // 7. Сбор результатов на процессе 0
  if (world_rank == 0) {
    GetOutput().resize(total_size);
    
    // Копируем данные процесса 0
    if (!local_data.empty()) {
      std::copy(local_data.begin(), local_data.end(), GetOutput().begin() + interval[0]);
    }
    
    // Получаем данные от остальных процессов
    for (int i = 1; i < world_size; ++i) {
      std::vector<int> other_interval = CalculateInterval(total_size, i, world_size);
      int other_size = other_interval[1] - other_interval[0] + 1;
      
      if (other_size > 0) {
        std::vector<int> other_data(other_size);
        MPI_Recv(other_data.data(), other_size, MPI_INT, i, 0, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::copy(other_data.begin(), other_data.end(), 
                  GetOutput().begin() + other_interval[0]);
      }
    }
  } else {
    // Отправляем данные процессу 0
    if (!local_data.empty()) {
      MPI_Send(local_data.data(), static_cast<int>(local_data.size()), 
               MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }
  
  return true;
}

bool ShkrebkoMShellSortBatcherMergeMPI::PostProcessingImpl() {
  // Распространяем результат на все процессы
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  int total_size = 0;
  if (world_rank == 0) {
    total_size = static_cast<int>(GetOutput().size());
  }
  
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (world_rank != 0) {
    GetOutput().resize(total_size);
  }
  
  MPI_Bcast(GetOutput().data(), total_size, MPI_INT, 0, MPI_COMM_WORLD);
  
  return !GetOutput().empty();
}

}  // namespace shkrebko_m_shell_sort_batcher_merge