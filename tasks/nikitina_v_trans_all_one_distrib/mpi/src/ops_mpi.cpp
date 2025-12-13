#include "nikitina_v_trans_all_one_distrib/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

#include "nikitina_v_trans_all_one_distrib/common/include/common.hpp"

namespace nikitina_v_trans_all_one_distrib {

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp = in;
  GetInput().swap(tmp);
}

bool TestTaskMPI::ValidationImpl() {
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  return true;
}

bool TestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int input_size = 0;
  if (rank == 0) {
    input_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&input_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (input_size == 0) {
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);

  int chunk_size = input_size / size;
  int remainder = input_size % size;

  for (int i = 0; i < size; ++i) {
    counts[i] = chunk_size + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }

  std::vector<int> local_input(counts[rank]);
  std::vector<int> local_output(counts[rank]);

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, counts.data(), displs.data(), MPI_INT, local_input.data(),
               counts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  std::copy(local_input.begin(), local_input.end(), local_output.begin());

  if (rank == 0) {
    GetOutput().resize(input_size);
  }

  MPI_Gatherv(local_output.data(), counts[rank], MPI_INT, rank == 0 ? GetOutput().data() : nullptr, counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_trans_all_one_distrib
