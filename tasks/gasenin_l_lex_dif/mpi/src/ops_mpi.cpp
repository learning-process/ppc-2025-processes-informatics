#include "gasenin_l_lex_dif/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gasenin_l_lex_dif/common/include/common.hpp"

namespace {

struct LocalDiff {
  size_t diff_pos;
  int result;
};

LocalDiff FindLocalDifference(const std::string &str1, const std::string &str2, size_t start, size_t end) {
  for (size_t i = start; i < end; ++i) {
    char c1 = (i < str1.length()) ? str1[i] : '\0';
    char c2 = (i < str2.length()) ? str2[i] : '\0';

    if (c1 != c2) {
      return {.diff_pos = i, .result = (c1 < c2) ? -1 : 1};
    }
  }
  return {.diff_pos = std::string::npos, .result = 0};
}

}  // namespace

namespace gasenin_l_lex_dif {

GaseninLLexDifMPI::GaseninLLexDifMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GaseninLLexDifMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &[str1, str2] = GetInput();
    return str1.length() <= 100000000 && str2.length() <= 100000000;
  }
  return true;
}

bool GaseninLLexDifMPI::PreProcessingImpl() {
  return true;
}

bool GaseninLLexDifMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &input_data = GetInput();
  auto &str1 = input_data.first;
  auto &str2 = input_data.second;

  std::vector<int> lengths(2, 0);

  if (rank == 0) {
    lengths[0] = str1.length();
    lengths[1] = str2.length();
  }

  MPI_Bcast(lengths.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    str1.resize(lengths[0]);
    str2.resize(lengths[1]);
  }

  if (lengths[0] > 0) {
    MPI_Bcast(str1.data(), static_cast<int>(lengths[0]), MPI_CHAR, 0, MPI_COMM_WORLD);
  }
  if (lengths[1] > 0) {
    MPI_Bcast(str2.data(), static_cast<int>(lengths[1]), MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  size_t total_len = std::max(str1.length(), str2.length());

  if (total_len == 0) {
    GetOutput() = 0;
    return true;
  }

  size_t chunk_size = (total_len + size - 1) / size;
  size_t start = rank * chunk_size;
  size_t end = std::min(start + chunk_size, total_len);

  if (start >= total_len) {
    start = total_len;
    end = total_len;
  }

  LocalDiff local_diff = FindLocalDifference(str1, str2, start, end);
  size_t local_diff_pos = (local_diff.diff_pos == std::string::npos) ? total_len : local_diff.diff_pos;
  int local_result = local_diff.result;

  int local_pos_int = static_cast<int>(local_diff_pos);
  int global_min_pos_int = 0;

  MPI_Allreduce(&local_pos_int, &global_min_pos_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  auto local_pos_64 = static_cast<uint64_t>(local_diff_pos);
  uint64_t global_min_pos_64 = 0;

  MPI_Allreduce(&local_pos_64, &global_min_pos_64, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);

  auto global_min_pos = static_cast<size_t>(global_min_pos_64);

  int result_for_sum = (local_diff_pos == global_min_pos) ? local_result : 0;

  int final_result = 0;
  MPI_Allreduce(&result_for_sum, &final_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (global_min_pos == total_len) {
    if (str1.length() != str2.length()) {
      final_result = (str1.length() < str2.length()) ? -1 : 1;
    } else {
      final_result = 0;
    }
  }

  GetOutput() = final_result;
  return true;
}

bool GaseninLLexDifMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gasenin_l_lex_dif
