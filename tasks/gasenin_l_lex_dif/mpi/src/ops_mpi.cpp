#include "gasenin_l_lex_dif/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "gasenin_l_lex_dif/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gasenin_l_lex_dif {

GaseninLLexDifMPI::GaseninLLexDifMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GaseninLLexDifMPI::ValidationImpl() {
  const auto &[str1, str2] = GetInput();
  return str1.length() <= 10000000 && str2.length() <= 10000000;
}

bool GaseninLLexDifMPI::PreProcessingImpl() {
  return true;
}

bool GaseninLLexDifMPI::RunImpl() {
  const auto &[str1, str2] = GetInput();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t total_len = std::max(str1.length(), str2.length());

  if (total_len == 0) {
    GetOutput() = 0;
    return true;
  }

  size_t chunk_size = (total_len + size - 1) / size;
  size_t start = rank * chunk_size;
  size_t end = std::min(start + chunk_size, total_len);

  int local_result = 0;
  size_t local_diff_pos = total_len;

  for (size_t i = start; i < end; ++i) {
    char c1 = (i < str1.length()) ? str1[i] : '\0';
    char c2 = (i < str2.length()) ? str2[i] : '\0';

    if (c1 != c2) {
      local_diff_pos = i;
      local_result = (c1 < c2) ? -1 : 1;
      break;
    }
  }

  uint64_t local_pos_64 = static_cast<uint64_t>(local_diff_pos);
  uint64_t global_min_pos_64 = 0;

  MPI_Allreduce(&local_pos_64, &global_min_pos_64, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);

  size_t global_min_pos = static_cast<size_t>(global_min_pos_64);

  int result_for_sum = 0;
  if (local_diff_pos == global_min_pos) {
    result_for_sum = local_result;
  }

  int final_result = 0;
  MPI_Allreduce(&result_for_sum, &final_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (global_min_pos == total_len) {
    if (str1.length() < str2.length()) {
      final_result = -1;
    } else if (str1.length() > str2.length()) {
      final_result = 1;
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
