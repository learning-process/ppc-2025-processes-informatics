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

  // 3. Проверка начала и конца чанка
  int starts_inside_word = 0, ends_inside_word = 0;
  if (local_size > 0) {
    starts_inside_word = !std::isspace((unsigned char)local[0]);
    ends_inside_word = !std::isspace((unsigned char)local.back());
  }

  std::vector<int> all_starts(world_size), all_ends(world_size);
  MPI_Allgather(&starts_inside_word, 1, MPI_INT, all_starts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&ends_inside_word, 1, MPI_INT, all_ends.data(), 1, MPI_INT, MPI_COMM_WORLD);

  size_t total_count = 0;
  MPI_Reduce(&local_count, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  // 4. Коррекция пересечений
  if (world_rank == 0) {
    for (int i = 1; i < world_size; ++i) {
      if (all_ends[i - 1] && all_starts[i]) {
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
