#include "shkrebko_m_count_char_freq/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "shkrebko_m_count_char_freq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkrebko_m_count_char_freq {

ShkrebkoMCountCharFreqMPI::ShkrebkoMCountCharFreqMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkrebkoMCountCharFreqMPI::ValidationImpl() {
  return (!std::get<0>(GetInput()).empty()) && (std::get<1>(GetInput()).length() == 1);
}

bool ShkrebkoMCountCharFreqMPI::PreProcessingImpl() {
  return true;
}

bool ShkrebkoMCountCharFreqMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string input_text = std::get<0>(GetInput());
  std::string target_char_str = std::get<1>(GetInput());
  char target_char = target_char_str[0];

  const int total_size = static_cast<int>(input_text.size());

  if (total_size == 0) {
    GetOutput() = 0;
    return true;
  }

  const int base = total_size / size;
  const int remainder = total_size % size;
  const int start = (rank * base) + std::min(rank, remainder);
  const int local_size = base + (rank < remainder ? 1 : 0);

  int local_count = 0;
  for (int i = start; i < start + local_size; ++i) {
    if (input_text[i] == target_char) {
      ++local_count;
    }
  }

  int global_result = 0;
  MPI_Reduce(&local_count, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;
  return true;
}

bool ShkrebkoMCountCharFreqMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkrebko_m_count_char_freq
