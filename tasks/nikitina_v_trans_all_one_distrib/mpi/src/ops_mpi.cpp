#include "nikitina_v_trans_all_one_distrib/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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
  if (GetInput().empty()) {
    return true;
  }

  GetOutput().resize(GetInput().size());

  auto size = static_cast<int>(GetInput().size());
  MPI_Reduce(GetInput().data(), GetOutput().data(), size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(GetOutput().data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_trans_all_one_distrib
