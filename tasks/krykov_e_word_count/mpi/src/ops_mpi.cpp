#include "krykov_e_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "krykov_e_word_count/common/include/common.hpp"

namespace krykov_e_word_count {

namespace {
void ComputeChunkSizesAndDispls(int text_size, int world_size, std::vector<int> &chunk_sizes,
                                std::vector<int> &displs) {
  int base_size = text_size / world_size;
  int remainder = text_size % world_size;

  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    chunk_sizes[i] = base_size + (i < remainder ? 1 : 0);
    displs[i] = offset;
    offset += chunk_sizes[i];
  }
}

uint64_t CountWordsInChunk(const std::vector<char> &local_chunk) {
  uint64_t local_count = 0;
  bool in_word = false;
  for (char c : local_chunk) {
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      in_word = false;
    } else {
      if (!in_word) {
        in_word = true;
        ++local_count;
      }
    }
  }
  return local_count;
}

std::pair<int, int> StartsEndsFromChunk(const std::vector<char> &local_chunk) {
  int starts_with_space = 1;
  int ends_with_space = 1;
  if (!local_chunk.empty()) {
    starts_with_space = (std::isspace(static_cast<unsigned char>(local_chunk.front())) != 0) ? 1 : 0;
    ends_with_space = (std::isspace(static_cast<unsigned char>(local_chunk.back())) != 0) ? 1 : 0;
  }
  return {starts_with_space, ends_with_space};
}

void AdjustTotalCountForBoundaries(const std::vector<uint64_t> &all_counts, const std::vector<int> &all_starts,
                                   const std::vector<int> &all_ends, uint64_t &total_count) {
  const std::size_t world_size = all_counts.size();
  for (std::size_t i = 1; i < world_size; ++i) {
    if (all_ends[i - 1] == 0 && all_starts[i] == 0) {
      if (total_count > 0) {
        --total_count;
      }
    }
  }
}

}  // namespace

KrykovEWordCountMPI::KrykovEWordCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrykovEWordCountMPI::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool KrykovEWordCountMPI::PreProcessingImpl() {
  const auto &input = GetInput();
  auto trimmed = input;

  trimmed.erase(trimmed.begin(),
                std::ranges::find_if(trimmed, [](char ch) { return !std::isspace(static_cast<unsigned char>(ch)); }));

  trimmed.erase(std::ranges::find_if(std::ranges::reverse_view(trimmed),
                                     [](char ch) {
    return !std::isspace(static_cast<unsigned char>(ch));
  }).base(),
                trimmed.end());

  GetInput() = trimmed;
  return true;
}

bool KrykovEWordCountMPI::RunImpl() {
  const std::string &text = GetInput();

  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (text.empty()) {
    GetOutput() = 0;
    return true;
  }

  int text_size = static_cast<int>(text.size());
  MPI_Bcast(&text_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> chunk_sizes(world_size);
  std::vector<int> displs(world_size);

  ComputeChunkSizesAndDispls(text_size, world_size, chunk_sizes, displs);

  int local_size = chunk_sizes[world_rank];
  std::vector<char> local_chunk(local_size);

  std::vector<char> text_buf;
  if (world_rank == 0) {
    text_buf.assign(text.begin(), text.end());
  }

  MPI_Scatterv(world_rank == 0 ? text_buf.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_CHAR,
               local_chunk.data(), local_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  uint64_t local_count = CountWordsInChunk(local_chunk);

  auto [starts_with_space, ends_with_space] = StartsEndsFromChunk(local_chunk);

  std::vector<int> all_starts(world_size);
  std::vector<int> all_ends(world_size);

  MPI_Gather(&starts_with_space, 1, MPI_INT, world_rank == 0 ? all_starts.data() : nullptr, 1, MPI_INT, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&ends_with_space, 1, MPI_INT, world_rank == 0 ? all_ends.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto local_count_ull = static_cast<uint64_t>(local_count);  // проблемная точка
  std::vector<uint64_t> all_counts_ull(world_size);
  MPI_Gather(&local_count_ull, 1, MPI_UINT64_T, world_rank == 0 ? all_counts_ull.data() : nullptr, 1, MPI_UINT64_T, 0,
             MPI_COMM_WORLD);

  if (world_rank == 0) {
    uint64_t total_count = 0;
    for (auto count_ull : all_counts_ull) {
      total_count += static_cast<uint64_t>(count_ull);
    }

    std::vector<uint64_t> all_counts(world_size);
    for (int i = 0; i < world_size; ++i) {
      all_counts[i] = static_cast<uint64_t>(all_counts_ull[i]);
    }

    AdjustTotalCountForBoundaries(all_counts, all_starts, all_ends, total_count);
    GetOutput() = static_cast<int>(total_count);
  }

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
