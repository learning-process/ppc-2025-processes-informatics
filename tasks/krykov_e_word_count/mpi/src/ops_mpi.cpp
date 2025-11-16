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
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool KrykovEWordCountMPI::PreProcessingImpl() {
  auto &input = GetInput();
  input.erase(input.begin(),
              std::find_if(input.begin(), input.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  input.erase(std::find_if(input.rbegin(), input.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              input.end());
  return true;
}

bool KrykovEWordCountMPI::RunImpl() {
  const std::string &text = GetInput();
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (text.empty()) {
    GetOutput() = 0;
    return true;
  }

  // Если процессов больше чем символов, считаем на процессе 0
  if (static_cast<size_t>(world_size) > text.size()) {
    if (world_rank == 0) {
      size_t count = 0;
      bool in_word = false;
      for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
          in_word = false;
        } else {
          if (!in_word) {
            in_word = true;
            count++;
          }
        }
      }
      GetOutput() = static_cast<int>(count);
    }
    MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  // Размер текста и его передача всем процессам
  int text_size = static_cast<int>(text.size());
  MPI_Bcast(&text_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Вычисление размеров чанков
  int base_size = text_size / world_size;
  int remainder = text_size % world_size;
  
  std::vector<int> chunk_sizes(world_size);
  std::vector<int> displs(world_size);
  
  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    chunk_sizes[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += chunk_sizes[i];
  }

  // Каждый процесс получает свой чанк
  int local_size = chunk_sizes[world_rank];
  std::vector<char> local_chunk(local_size);
  
  MPI_Scatterv(world_rank == 0 ? text.data() : nullptr,
               chunk_sizes.data(), displs.data(), MPI_CHAR,
               local_chunk.data(), local_size, MPI_CHAR,
               0, MPI_COMM_WORLD);

  // Локальный подсчет слов
  size_t local_count = 0;
  bool in_word = false;
  for (char c : local_chunk) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      in_word = false;
    } else {
      if (!in_word) {
        in_word = true;
        local_count++;
      }
    }
  }

  // Сбор информации о границах для коррекции
  // Важно: анализируем реальные символы на границах
  bool starts_with_letter = false;
  bool ends_with_letter = false;
  
  if (local_size > 0) {
    starts_with_letter = !std::isspace(static_cast<unsigned char>(local_chunk[0]));
    ends_with_letter = !std::isspace(static_cast<unsigned char>(local_chunk[local_size - 1]));
  }

  // Преобразуем bool в int для MPI
  int starts_int = starts_with_letter ? 1 : 0;
  int ends_int = ends_with_letter ? 1 : 0;

  std::vector<int> all_starts(world_size);
  std::vector<int> all_ends(world_size);
  
  MPI_Gather(&starts_int, 1, MPI_INT, 
             world_rank == 0 ? all_starts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&ends_int, 1, MPI_INT, 
             world_rank == 0 ? all_ends.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Сбор локальных счетчиков
  std::vector<size_t> all_counts(world_size);
  MPI_Gather(&local_count, 1, MPI_UNSIGNED_LONG_LONG,
             world_rank == 0 ? all_counts.data() : nullptr, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // Коррекция на процессе 0
  size_t total_count = 0;
  if (world_rank == 0) {
    // Суммируем все локальные счетчики
    for (size_t count : all_counts) {
      total_count += count;
    }
    
    // Коррекция разделенных слов
    // Если предыдущий чанк заканчивается буквой И текущий начинается с буквы,
    // значит слово было разделено и посчитано дважды
    for (int i = 1; i < world_size; ++i) {
      if (all_ends[i - 1] == 1 && all_starts[i] == 1) {
        total_count--;
      }
    }
  }

  // Распространение результата на все процессы
  size_t global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    // Применяем коррекцию к глобальному счетчику
    for (int i = 1; i < world_size; ++i) {
      if (all_ends[i - 1] == 1 && all_starts[i] == 1) {
        global_count--;
      }
    }
    GetOutput() = static_cast<int>(global_count);
  }

  // Распространяем финальный результат
  int result = 0;
  if (world_rank == 0) {
    result = GetOutput();
  }
  MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (world_rank != 0) {
    GetOutput() = result;
  }

  return true;
}

bool KrykovEWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_word_count
