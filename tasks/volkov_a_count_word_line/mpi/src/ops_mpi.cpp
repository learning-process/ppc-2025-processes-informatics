#include "volkov_a_count_word_line/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

#include "volkov_a_count_word_line/common/include/common.hpp"

namespace volkov_a_count_word_line {

namespace {

bool IsTokenChar(char c) {
  const bool is_alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  const bool is_digit = (c >= '0' && c <= '9');
  const bool is_special = (c == '-' || c == '_');
  return is_alpha || is_digit || is_special;
}

int CountWordsInChunk(const std::vector<char> &data, int valid_size) {
  int count = 0;
  bool in_word = false;

  for (int i = 0; i < valid_size; ++i) {
    if (IsTokenChar(data[i])) {
      if (!in_word) {
        in_word = true;
        count++;
      }
    } else {
      in_word = false;
    }
  }

  if (in_word && static_cast<size_t>(valid_size) < data.size() && IsTokenChar(data[valid_size])) {
    count--;
  }

  return count;
}

int CountWordsSeq(const std::string &str) {
  int count = 0;
  bool in_word = false;
  for (char c : str) {
    if (IsTokenChar(c)) {
      if (!in_word) {
        in_word = true;
        count++;
      }
    } else {
      in_word = false;
    }
  }
  return count;
}

}  // namespace

VolkovACountWordLineMPI::VolkovACountWordLineMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VolkovACountWordLineMPI::ValidationImpl() {
  return true;
}

bool VolkovACountWordLineMPI::PreProcessingImpl() {
  return true;
}

bool VolkovACountWordLineMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::string &input_str = GetInput();

  if (input_str.empty()) {
    GetOutput() = 0;
    return true;
  }

  if (static_cast<size_t>(world_size) > input_str.size()) {
    if (rank == 0) {
      GetOutput() = CountWordsSeq(input_str);
    }
    MPI_Bcast(&GetOutput(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    return true;
  }

  size_t remainder = input_str.size() % static_cast<size_t>(world_size);
  size_t padding = (static_cast<size_t>(world_size) - remainder) % static_cast<size_t>(world_size);
  input_str.append(padding + static_cast<size_t>(world_size), ' ');

  int chunk_size = static_cast<int>(input_str.size() / static_cast<size_t>(world_size));

  std::vector<int> send_counts(world_size);
  std::vector<int> displs(world_size);

  if (rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      send_counts[i] = chunk_size + 1;
      displs[i] = i * chunk_size;
    }
  }

  std::vector<char> local_data(chunk_size + 1);

  MPI_Scatterv(input_str.data(), send_counts.data(), displs.data(), MPI_CHAR, local_data.data(), chunk_size + 1,
               MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_words = CountWordsInChunk(local_data, chunk_size);
  int total_words = 0;

  MPI_Allreduce(&local_words, &total_words, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = total_words;

  return true;
}

bool VolkovACountWordLineMPI::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_count_word_line
