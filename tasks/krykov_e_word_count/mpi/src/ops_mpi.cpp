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

size_t CountLocalWords(const std::vector<char> &local_buf, int part) {
  size_t local_cnt = 0;
  bool in_word = false;

  // Подсчет слов в основной части
  for (int i = 0; i < part; i++) {
    if (IsWordChar(local_buf[i])) {
      if (!in_word) {
        in_word = true;
        local_cnt++;
      }
    } else {
      in_word = false;
    }
  }

  // Коррекция: если слово продолжается в следующем чанке, вычитаем 1
  if (in_word && IsWordChar(local_buf[part])) {
    local_cnt--;
  }

  return local_cnt;
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
      GetOutput() = count;
    }
    // Распространяем результат на все процессы
    MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  // Дополняем текст пробелами до кратности world_size
  std::string padded_text = text;
  size_t rem = padded_text.size() % static_cast<size_t>(world_size);
  if (rem != 0) {
    padded_text.append(world_size - rem, ' ');
  }

  size_t part = padded_text.size() / static_cast<size_t>(world_size);

  // Каждый процесс получает part + 1 символов для перекрытия
  std::vector<int> send_counts(world_size, static_cast<int>(part + 1));
  std::vector<int> displs(world_size);

  if (world_rank == 0) {
    for (int i = 0; i < world_size; i++) {
      displs[i] = static_cast<int>(i * part);
    }
  }

  std::vector<char> local_buf(part + 1);

  // Распределяем данные с перекрытием
  MPI_Scatterv(world_rank == 0 ? padded_text.data() : nullptr, send_counts.data(), displs.data(), MPI_CHAR,
               local_buf.data(), static_cast<int>(part + 1), MPI_CHAR, 0, MPI_COMM_WORLD);

  // Локальный подсчет с коррекцией
  size_t local_cnt = CountLocalWords(local_buf, static_cast<int>(part));

  // Суммируем результаты на всех процессах
  size_t global_cnt = 0;
  MPI_Allreduce(&local_cnt, &global_cnt, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<int>(global_cnt);

  return true;
}

bool KrykovEWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_word_count
