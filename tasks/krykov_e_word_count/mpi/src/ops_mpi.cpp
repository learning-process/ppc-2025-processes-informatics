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
  // return (!GetInput().empty()) && (GetOutput() == 0);
  return true;
}

bool KrykovEWordCountMPI::PreProcessingImpl() {
  auto &input = GetInput();
  input.erase(input.begin(),
              std::find_if(input.begin(), input.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  input.erase(std::find_if(input.rbegin(), input.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
              input.end());
  return true;
}

namespace {

bool IsWordChar(char c) {
  return !std::isspace(static_cast<unsigned char>(c));
}

/*size_t CountLocalWords(const std::vector<char> &local_buf, int part_size) {
  size_t count = 0;
  bool in_word = false;

  // Считаем слова только в основной части (part_size символов)
  for (int i = 0; i < part_size; i++) {
    if (IsWordChar(local_buf[i])) {
      if (!in_word) {
        in_word = true;
        count++;
      }
    } else {
      in_word = false;
    }
  }*/

// Коррекция: если слово продолжается в дополнительном символе, значит мы его разделили
if (in_word && IsWordChar(local_buf[part_size])) {
  count--;
}

return count;
}

}  // namespace

bool KrykovEWordCountMPI::RunImpl() {
  const std::string &text = GetInput();
  int world_size = 0, world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (text.empty()) {
    GetOutput() = 0;
    return true;
  }

  // Простая обработка для небольшого количества процессов
  if (world_size == 1) {
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

  MPI_Scatterv(world_rank == 0 ? text.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_CHAR, local_chunk.data(),
               local_size, MPI_CHAR, 0, MPI_COMM_WORLD);

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
  int starts_with_space = local_size > 0 ? (std::isspace(static_cast<unsigned char>(local_chunk[0])) ? 1 : 0) : 1;
  int ends_with_space =
      local_size > 0 ? (std::isspace(static_cast<unsigned char>(local_chunk[local_size - 1])) ? 1 : 0) : 1;

  std::vector<int> all_starts(world_size);
  std::vector<int> all_ends(world_size);

  MPI_Gather(&starts_with_space, 1, MPI_INT, world_rank == 0 ? all_starts.data() : nullptr, 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&ends_with_space, 1, MPI_INT, world_rank == 0 ? all_ends.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Сбор локальных счетчиков
  std::vector<size_t> all_counts(world_size);
  MPI_Gather(&local_count, 1, MPI_UNSIGNED_LONG_LONG, world_rank == 0 ? all_counts.data() : nullptr, 1,
             MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // Коррекция на процессе 0
  if (world_rank == 0) {
    size_t total_count = 0;
    for (size_t count : all_counts) {
      total_count += count;
    }

    // Коррекция разделенных слов
    for (int i = 1; i < world_size; ++i) {
      // Если предыдущий чанк заканчивается на не-пробел и текущий начинается с не-пробела,
      // значит слово разделено между процессами
      if (all_ends[i - 1] == 0 && all_starts[i] == 0) {
        total_count--;
      }
    }

    GetOutput() = static_cast<int>(total_count);
  }

  // Распространение результата на все процессы
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
