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

  size_t text_size = text.size();
  MPI_Bcast(&text_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  if (text_size == 0) {
    if (world_rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  // 1. Разбиваем на чанки
  std::vector<int> chunk_sizes(world_size, 0), displs(world_size, 0);
  if (world_rank == 0) {
    int base = text_size / world_size;
    int rem = text_size % world_size;
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
      chunk_sizes[i] = base + (i < rem ? 1 : 0);
      displs[i] = offset;
      offset += chunk_sizes[i];
    }
  }
  MPI_Bcast(chunk_sizes.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_size = chunk_sizes[world_rank];
  std::string local(local_size, '\0');
  MPI_Scatterv(world_rank == 0 ? text.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_CHAR, local.data(),
               local_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  // 2. Подсчёт слов локально
  size_t local_count = 0;
  bool in_word = false;
  for (char c : local) {
    if (std::isspace((unsigned char)c)) {
      in_word = false;
    } else if (!in_word) {
      in_word = true;
      local_count++;
    }
  }

  // 3. Определение состояния границ для коррекции
  // Получаем первый и последний символ соседних чанков для корректной обработки границ
  char first_char = local_size > 0 ? local[0] : ' ';
  char last_char = local_size > 0 ? local[local_size - 1] : ' ';

  // Собираем последние символы всех чанков (кроме последнего)
  std::vector<char> prev_chars(world_size, ' ');
  MPI_Allgather(&last_char, 1, MPI_CHAR, prev_chars.data(), 1, MPI_CHAR, MPI_COMM_WORLD);

  // Собираем первые символы всех чанков (кроме первого)
  std::vector<char> next_chars(world_size, ' ');
  MPI_Allgather(&first_char, 1, MPI_CHAR, next_chars.data(), 1, MPI_CHAR, MPI_COMM_WORLD);

  // 4. Подсчет общего количества и коррекция
  size_t total_count = 0;
  MPI_Reduce(&local_count, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    // Корректируем слова, которые были разделены между процессами
    for (int i = 1; i < world_size; ++i) {
      // Если последний символ предыдущего чанка не пробел и первый символ текущего чанка не пробел,
      // значит слово было разделено между процессами и мы его посчитали дважды
      if (!std::isspace((unsigned char)prev_chars[i - 1]) && !std::isspace((unsigned char)next_chars[i])) {
        total_count--;
      }
    }
    GetOutput() = total_count;
  }
  return true;
}

bool KrykovEWordCountMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace krykov_e_word_count
