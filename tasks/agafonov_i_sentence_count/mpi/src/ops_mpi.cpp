#include "agafonov_i_sentence_count/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <cctype>
#include <vector>

#include <mpi.h>

namespace agafonov_i_sentence_count {

SentenceCountMPI::SentenceCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SentenceCountMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool SentenceCountMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SentenceCountMPI::RunImpl() {
  const std::string& text = GetInput();
  
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int total_length = static_cast<int>(text.length());
  int chunk_size = total_length / world_size;
  int remainder = total_length % world_size;

  int start = world_rank * chunk_size + std::min(world_rank, remainder);
  int end = start + chunk_size + (world_rank < remainder ? 1 : 0);

  // Локальный подсчет
  int local_count = 0;
  bool local_in_sentence = false;

  // Учитываем контекст с предыдущего чанка
  if (world_rank > 0 && start > 0) {
    char prev_char = text[start - 1];
    local_in_sentence = std::isalpha(prev_char) || std::isdigit(prev_char);
  }

  for (int i = start; i < end && i < total_length; ++i) {
    char c = text[i];
    
    if (std::isalpha(static_cast<unsigned char>(c)) || std::isdigit(static_cast<unsigned char>(c))) {
      local_in_sentence = true;
    } else if (c == '.' && local_in_sentence) {
      // Проверяем, что это не многоточие
      if (i + 1 < total_length && text[i + 1] == '.') {
        // Это часть многоточия, пропускаем
        continue;
      }
      local_count++;
      local_in_sentence = false;
    } else if ((c == '!' || c == '?') && local_in_sentence) {
      local_count++;
      local_in_sentence = false;
    }
  }

  // Собираем результаты
  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Корректировка на главном процессе
  if (world_rank == 0) {
    // Проверяем границы между чанками
    for (int i = 1; i < world_size; ++i) {
      int chunk_end = (i * chunk_size + std::min(i, remainder)) - 1;
      if (chunk_end >= 0 && chunk_end < total_length - 1) {
        char end_char = text[chunk_end];
        char start_char = text[chunk_end + 1];
        
        // Если предложение прервано на границе чанков
        if ((std::isalpha(static_cast<unsigned char>(end_char)) || 
             std::isdigit(static_cast<unsigned char>(end_char))) && 
            start_char == '.') {
          global_count++;
        }
      }
    }

    // Учитываем незавершенное предложение в конце
    if (total_length > 0) {
      char last_char = text[total_length - 1];
      if (std::isalpha(static_cast<unsigned char>(last_char)) || 
          std::isdigit(static_cast<unsigned char>(last_char))) {
        global_count++;
      }
    }

    GetOutput() = global_count;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool SentenceCountMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace agafonov_i_sentence_count