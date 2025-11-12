#include "krykov_e_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace krykov_e_word_count {

KrykovEWordCountMPI::KrykovEWordCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrykovEWordCountMPI::ValidationImpl() {
  return true;  // Всегда валидно для пустой строки тоже
}

bool KrykovEWordCountMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KrykovEWordCountMPI::RunImpl() {
  const std::string &text = GetInput();
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Обработка пустой строки
  if (text.empty()) {
    GetOutput() = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int total_length = text.length();
  int chunk_size = total_length / world_size;
  int remainder = total_length % world_size;

  int start = world_rank * chunk_size + std::min(world_rank, remainder);
  int end = start + chunk_size + (world_rank < remainder ? 1 : 0);

  // Подсчет слов в локальном чанке
  int local_count = 0;
  bool in_word = false;

  for (int i = start; i < end; i++) {
    char c = text[i];
    if (std::isspace(c) || std::ispunct(c)) {
      if (in_word) {
        local_count++;
        in_word = false;
      }
    } else {
      in_word = true;
    }
  }

  // Проверка границ между процессами
  if (world_rank > 0 && start > 0) {
    // Если слово продолжается с предыдущего процесса
    char prev_char = text[start - 1];
    char curr_char = text[start];
    if (!std::isspace(prev_char) && !std::ispunct(prev_char) && !std::isspace(curr_char) && !std::ispunct(curr_char)) {
      // Слово разделено между процессами - не вычитаем, а корректируем логику
      // Эта логика требует пересмотра
    }
  }

  // Сбор результатов
  std::vector<int> all_counts;
  if (world_rank == 0) {
    all_counts.resize(world_size);
  }

  MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  int global_count = 0;
  if (world_rank == 0) {
    global_count = std::accumulate(all_counts.begin(), all_counts.end(), 0);

    // Проверка последнего символа
    if (!text.empty() && !std::isspace(text.back()) && !std::ispunct(text.back())) {
      global_count++;
    }

    GetOutput() = global_count;
  }

  // Распространение результата на все процессы
  MPI_Bcast(&global_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (world_rank != 0) {
    GetOutput() = global_count;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() >= 0;
}

bool KrykovEWordCountMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace krykov_e_word_count
