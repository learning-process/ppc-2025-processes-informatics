#include "potashnik_m_char_freq/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>
#include <string>

#include "potashnik_m_char_freq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace potashnik_m_char_freq {

PotashnikMCharFreqMPI::PotashnikMCharFreqMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PotashnikMCharFreqMPI::ValidationImpl() {
  return (std::get<0>(GetInput()).size() > 0);
}

bool PotashnikMCharFreqMPI::PreProcessingImpl() {
  return true;
}

bool PotashnikMCharFreqMPI::RunImpl() {
  auto &input = GetInput();
  std::string str = std::get<0>(input);
  char chr = std::get<1>(input);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int string_size = static_cast<int>(str.size());
  int block_size = string_size / world_size;
  
  int start_pos = block_size * rank;
  int end_pos;
  if (rank == world_size - 1) {
    end_pos = string_size;
  }
  else {
    end_pos = start_pos + block_size;
  }
  int cur_res = 0;
  
  for (int i = start_pos; i < end_pos; i++) {
    if (str[i] == chr) cur_res++;
  }

  int total_res = 0;
  MPI_Allreduce(&cur_res, &total_res, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = total_res;

  return true;
}

bool PotashnikMCharFreqMPI::PostProcessingImpl() {
  return true;
}

}  // namespace potashnik_m_char_freq
