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
  int world_size = 0;
  int world_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // ------------------------------
  // 1. Проверка пустого ввода
  // ------------------------------
  size_t text_size = text.size();
  MPI_Bcast(&text_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  if (text_size == 0) {
    if (world_rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  // ------------------------------
  // 2. Раздача размеров чанков
  // ------------------------------
  std::vector<int> chunk_sizes(world_size, 0);
  std::vector<int> displs(world_size, 0);

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

  // Рассылаем размеры
  MPI_Bcast(chunk_sizes.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

  // ------------------------------
  // 3. Получаем локальный чанк
  // ------------------------------
  int local_size = chunk_sizes[world_rank];
  std::string local(local_size, '\0');

  MPI_Scatterv(world_rank == 0 ? text.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_CHAR, local.data(),
               local_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  // ------------------------------
  // 4. Локальный подсчёт слов
  // ------------------------------
  bool in_word = false;
  size_t local_count = 0;

  for (char c : local) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (in_word) {
        in_word = false;
      }
    } else {
      if (!in_word) {
        in_word = true;
        local_count++;
      }
    }
  }

  // ------------------------------------------------------------
  // 5. Коррекция границы между чанками
  // ------------------------------------------------------------
  // Если наш чанк начинается НЕ с начала текста
  // нужно спросить у предыдущего чанка, заканчивался ли он словом

  int starts_with_word = 0;  // 1 если local[0] не пробел и мы в начале слова
  if (local_size > 0 && !std::isspace(static_cast<unsigned char>(local[0]))) {
    starts_with_word = 1;
  }

  // Узнать, заканчивался ли предыдущий блок словом
  int prev_ended_in_word = 0;

  if (world_rank > 0) {
    // Получаем флаг от предыдущего процесса
    MPI_Recv(&prev_ended_in_word, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Проверяем, заканчиваемся ли мы словом
  int ended_in_word = in_word ? 1 : 0;

  // Посылаем свой ended_in_word следующему процессу
  if (world_rank < world_size - 1) {
    MPI_Send(&ended_in_word, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
  }

  // Если начало нашего чанка выглядит как начало слова,
  // но предыдущий чанк заканчивался словом — значит мы *не начинаем новое слово*
  if (starts_with_word && prev_ended_in_word) {
    local_count--;
  }

  // ------------------------------
  // 6. Суммирование результата
  // ------------------------------
  size_t total_count = 0;
  MPI_Reduce(&local_count, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = total_count;
  }

  return true;
}

bool KrykovEWordCountMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace krykov_e_word_count
