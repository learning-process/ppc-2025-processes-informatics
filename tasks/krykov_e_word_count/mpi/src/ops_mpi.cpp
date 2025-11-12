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
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::string &input = GetInput();
  const size_t total_length = input.size();

  // Разделяем строку на подстроки для каждого процесса
  size_t chunk_size = total_length / size;
  size_t start = rank * chunk_size;
  size_t end = (rank == size - 1) ? total_length : start + chunk_size;

  // Корректируем границы, чтобы не разорвать слова
  // Если не первый процесс и текущий символ не пробел, отодвигаем начало до следующего пробела
  if (rank != 0 && start < total_length) {
    while (start < total_length && !std::isspace(static_cast<unsigned char>(input[start]))) {
      start++;
    }
  }
  // Если не последний процесс и не дошли до конца, продлеваем кусок до ближайшего пробела
  if (rank != size - 1 && end < total_length) {
    while (end < total_length && !std::isspace(static_cast<unsigned char>(input[end]))) {
      end++;
    }
  }

  // Подстрока для данного процесса
  std::string local_str = input.substr(start, end - start);

  // Подсчёт слов в локальной части
  size_t local_count = 0;
  bool in_word = false;
  for (char ch : local_str) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
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

  // Суммируем количество слов со всех процессов
  size_t global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_count;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KrykovEWordCountMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace krykov_e_word_count
