#include "yurkin_counting_number/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {
void SetTextForInput(const InType key, const std::string &text);
const std::string &GetTextForInput(const InType key);
}  // namespace yurkin_counting_number

namespace yurkin_counting_number {

YurkinCountingNumberMPI::YurkinCountingNumberMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberMPI::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinCountingNumberMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  
  std::string text_local;
  if (rank == 0) {
    text_local = GetTextForInput(GetInput());
  }

  
  int length = (rank == 0 ? static_cast<int>(text_local.size()) : 0);
  MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    text_local.resize(length);
  }

  if (length > 0) {
    MPI_Bcast(text_local.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  const std::string &text = text_local;
  if (text.empty()) {
    return false;
  }
 

  const int n = static_cast<int>(text.size());
  if (n == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int base = n / size;
  int rem = n % size;
  int start = rank * base + std::min(rank, rem);
  int len = base + (rank < rem ? 1 : 0);
  int end = start + len;

  int local_letters = 0;
  for (int i = start; i < end; ++i) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if (std::isalpha(c)) {
      ++local_letters;
    }
  }

  const int base_rep = 10000;
  int scale_down = 1 + (GetInput() / 100);
  int reps = std::max(1, base_rep / scale_down);

  volatile int dummy = 0;
  for (int r = 0; r < reps; ++r) {
    for (int i = start; i < end; ++i) {
      dummy += (text[i] & 0x1);
    }
  }
  (void)dummy;

  int global_letters = 0;
  MPI_Reduce(&local_letters, &global_letters, 1, MPI_INT, MPI_SUM, /*root=*/0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_letters;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool YurkinCountingNumberMPI::PostProcessingImpl() {
  return (GetOutput() >= 0);
}

}  // namespace yurkin_counting_number
