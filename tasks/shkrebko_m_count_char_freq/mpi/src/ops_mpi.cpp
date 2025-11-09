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
  const auto& input_data = GetInput();
  return !input_data.first.empty(); 
}

bool ShkrebkoMCountCharFreqMPI::PreProcessingImpl() {
  const auto& input_data = GetInput();
  input_text_ = input_data.first;      
  target_char_ = input_data.second;    
  return true;
}

bool ShkrebkoMCountCharFreqMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_size = static_cast<int>(input_text_.size());
  
  if (total_size == 0) {
    global_result_ = 0;
    GetOutput() = global_result_;
    return true;
  }


  const int base = total_size / size;
  const int remainder = total_size % size;
  const int start = (rank * base) + std::min(rank, remainder);
  const int local_size = base + (rank < remainder ? 1 : 0);


  int local_count = 0;
  for (int i = start; i < start + local_size; ++i) {
    if (input_text_[i] == target_char_) {
      ++local_count;
    }
  }


  MPI_Reduce(&local_count, &global_result_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_result_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_result_;
  return true;
}

bool ShkrebkoMCountCharFreqMPI::PostProcessingImpl() {
  return true;  
}

}  // namespace shkrebko_m_count_char_freq