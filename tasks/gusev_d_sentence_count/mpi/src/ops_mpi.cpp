#include "gusev_d_sentence_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "gusev_d_sentence_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace gusev_d_sentence_count {

static bool IsTerminator(char c) {
  return c == '.' || c == '!' || c == '?';
}

GusevDSentenceCountMPI::GusevDSentenceCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GusevDSentenceCountMPI::ValidationImpl() {
  return true;
}

bool GusevDSentenceCountMPI::PreProcessingImpl() {
  return true;
}

static size_t CountSentencesInChunk(const std::vector<char> &local_chunk, int chunk_size) {
  size_t sentence_count = 0;
  
  for (int i = 0; i < chunk_size; ++i) {
    if (IsTerminator(local_chunk[i])) {
      if (!IsTerminator(local_chunk[i + 1])) {
        sentence_count++;
      }
    }
  }
  return sentence_count;
}

/*bool GusevDSentenceCountMPI::RunImpl() {
  auto input = GetInput();
  if (input == 0) {
    return false;
  }

  for (InType i = 0; i < GetInput(); i++) {
    for (InType j = 0; j < GetInput(); j++) {
      for (InType k = 0; k < GetInput(); k++) {
        std::vector<InType> tmp(i + j + k, 1);
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
        GetOutput() -= i + j + k;
      }
    }
  }

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() /= num_threads;
  } else {
    int counter = 0;
    for (int i = 0; i < num_threads; i++) {
      counter++;
    }

    if (counter != 0) {
      GetOutput() /= counter;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() > 0;
}*/

bool GusevDSentenceCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace gusev_d_sentence_count
