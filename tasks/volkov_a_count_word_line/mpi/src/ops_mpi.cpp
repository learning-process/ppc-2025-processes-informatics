#include "volkov_a_count_word_line/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
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

int CountWordsInChunk(const std::vector<char> &data, char prev_char) {
  if (data.empty()) {
    return 0;
  }

  enum class State : std::uint8_t { kInWord, kInSeparator };
  int word_count = 0;

  State current_state = IsTokenChar(prev_char) ? State::kInWord : State::kInSeparator;

  for (const char c : data) {
    if (current_state == State::kInSeparator) {
      if (IsTokenChar(c)) {
        word_count++;
        current_state = State::kInWord;
      }
    } else {
      if (!IsTokenChar(c)) {
        current_state = State::kInSeparator;
      }
    }
  }

  return word_count;
}

void PrepareScatterVArgs(int total_len, int world_size, std::vector<int> *send_counts, std::vector<int> *displs) {
  const int base_chunk = total_len / world_size;
  const int remainder = total_len % world_size;
  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    (*send_counts)[i] = base_chunk + (i < remainder ? 1 : 0);
    (*displs)[i] = offset;
    offset += (*send_counts)[i];
  }
}

char ExchangeBoundaryChars(int rank, int world_size, const std::vector<char> &local_data) {
  char prev_char = ' ';
  char my_last_char = local_data.empty() ? ' ' : local_data.back();
  const int left_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  const int right_neighbor = (rank == world_size - 1) ? MPI_PROC_NULL : rank + 1;

  MPI_Sendrecv(&my_last_char, 1, MPI_CHAR, right_neighbor, 0, &prev_char, 1, MPI_CHAR, left_neighbor, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  return prev_char;
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

  int total_len = 0;
  if (rank == 0) {
    total_len = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_len == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  const int base_chunk = total_len / world_size;
  const int remainder = total_len % world_size;
  const int my_count = base_chunk + (rank < remainder ? 1 : 0);

  std::vector<int> send_counts(world_size);
  std::vector<int> displs(world_size);
  if (rank == 0) {
    PrepareScatterVArgs(total_len, world_size, &send_counts, &displs);
  }

  std::vector<char> local_data(my_count);
  const char *send_buf = (rank == 0) ? GetInput().data() : nullptr;

  MPI_Scatterv(send_buf, (rank == 0 ? send_counts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_CHAR,
               local_data.data(), my_count, MPI_CHAR, 0, MPI_COMM_WORLD);

  char prev_char = ExchangeBoundaryChars(rank, world_size, local_data);

  int local_words = CountWordsInChunk(local_data, prev_char);
  int total_words = 0;
  MPI_Reduce(&local_words, &total_words, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = total_words;
  }

  return true;
}

bool VolkovACountWordLineMPI::PostProcessingImpl() {
  return true;
}

}  // namespace volkov_a_count_word_line
