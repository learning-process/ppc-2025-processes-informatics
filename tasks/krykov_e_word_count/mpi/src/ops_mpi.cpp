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

size_t CountLocalWords(const std::vector<char> &local_buf, int part_size) {
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
  }

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

  // Если процессов больше чем символов, считаем на процессе 0
  if (static_cast<size_t>(world_size) > text.size()) {
    if (world_rank == 0) {
      size_t count = 0;
      bool in_word = false;
      for (char c : text) {
        if (IsWordChar(c)) {
          if (!in_word) {
            in_word = true;
            count++;
          }
        } else {
          in_word = false;
        }
      }
      GetOutput() = static_cast<int>(count);
    }
    MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  // Дополняем текст пробелами до нужного размера
  std::string padded_text = text;
  size_t text_size = text.size();
  size_t part = (text_size + world_size - 1) / world_size;  // ceil division
  size_t total_size = part * world_size;

  // Добавляем один дополнительный символ в конец для последнего процесса
  padded_text.append(total_size - text_size + 1, ' ');

  // Распределение данных: каждый процесс получает part символов
  // Но последний процесс получает part + 1 символов для перекрытия
  std::vector<int> send_counts(world_size);
  std::vector<int> displs(world_size);

  for (int i = 0; i < world_size; i++) {
    displs[i] = static_cast<int>(i * part);
    send_counts[i] = static_cast<int>(part);
  }

  // Последний процесс получает на 1 символ больше для перекрытия
  send_counts[world_size - 1] = static_cast<int>(part + 1);

  // Каждый процесс готовится принять максимальное количество символов
  int recv_count = (world_rank == world_size - 1) ? static_cast<int>(part + 1) : static_cast<int>(part);
  std::vector<char> local_buf(recv_count);

  // Распределяем данные
  MPI_Scatterv(world_rank == 0 ? padded_text.data() : nullptr, send_counts.data(), displs.data(), MPI_CHAR,
               local_buf.data(), recv_count, MPI_CHAR, 0, MPI_COMM_WORLD);

  // Локальный подсчет: используем part символов для всех процессов
  size_t local_count = CountLocalWords(local_buf, static_cast<int>(part));

  // Суммируем результаты
  size_t global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<int>(global_count);

  return true;
}

bool KrykovEWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_word_count
