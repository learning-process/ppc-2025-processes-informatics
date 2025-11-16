#include "zavyalov_a_scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include "util/include/util.hpp"
#include "zavyalov_a_scalar_product/common/include/common.hpp"

namespace zavyalov_a_scalar_product {

ZavyalovAScalarProductMPI::ZavyalovAScalarProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ZavyalovAScalarProductMPI::ValidationImpl() {
  return (std::get<0>(GetInput()).size() > 0) && (std::get<0>(GetInput()).size() == std::get<1>(GetInput()).size());
}

bool ZavyalovAScalarProductMPI::PreProcessingImpl() {
  return true;
}

bool ZavyalovAScalarProductMPI::RunImpl() {
  auto &input = GetInput();
  const std::vector<double> &left = std::get<0>(input);
  const std::vector<double> &right = std::get<1>(input);

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int vectorSize = left.size();

  int blocksize = vectorSize / worldSize;
  int elementsLeft = vectorSize - worldSize * blocksize;

  double curRes = 0.0;
  int start = blocksize * rank + std::min(rank, elementsLeft);
  int end = start + blocksize + (rank < elementsLeft ? 1 : 0);
  for (int i = start; i < end; i++) {
    curRes += left[i] * right[i];
  }

  double globRes = 0.0;
  MPI_Allreduce(&curRes, &globRes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = globRes;
  return true;
}

bool ZavyalovAScalarProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_scalar_product
