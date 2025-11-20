#include "zavyalov_a_scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "zavyalov_a_scalar_product/common/include/common.hpp"

namespace zavyalov_a_scalar_product {

ZavyalovAScalarProductMPI::ZavyalovAScalarProductMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = 0.0;
}

bool ZavyalovAScalarProductMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  return (!std::get<0>(GetInput()).empty()) && (std::get<0>(GetInput()).size() == std::get<1>(GetInput()).size());
}

bool ZavyalovAScalarProductMPI::PreProcessingImpl() {
  return true;
}
bool ZavyalovAScalarProductMPI::RunImpl() {
  std::vector<double> left;
  std::vector<double> right;

  // int worldSize = ppc::util::GetNumThreads();
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int vectorSize = 0;

  if (rank == 0) {
    GetOutput() = 0.0;
    const auto &input = GetInput();
    if (!std::get<0>(input).empty()) {  // it does not compile in ubuntu without this line
      left = std::get<0>(input);
    }
    if (!std::get<1>(input).empty()) {  // it does not compile in ubuntu without this line
      right = std::get<1>(input);
    }
    vectorSize = left.size();
  }

  MPI_Bcast(&vectorSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> sendcounts(worldSize);
  std::vector<int> displs(worldSize);

  int blocksize = vectorSize / worldSize;
  int elementsLeft = vectorSize - worldSize * blocksize;
  int elements_processed = 0;

  for (int i = 0; i < worldSize; i++) {
    sendcounts[i] = blocksize + (i < elementsLeft ? 1 : 0);
    displs[i] = elements_processed;
    elements_processed += sendcounts[i];
  }

  int elements_count = sendcounts[rank];
  std::vector<double> local_left(elements_count);
  std::vector<double> local_right(elements_count);

  MPI_Scatterv(left.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_left.data(), elements_count, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
  MPI_Scatterv(right.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_right.data(), elements_count,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double curRes = 0.0;
  for (int i = 0; i < elements_count; i++) {
    curRes += local_left[i] * local_right[i];
  }

  double globRes = 0.0;
  MPI_Reduce(&curRes, &globRes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  GetOutput() = globRes;

  return true;
}

bool ZavyalovAScalarProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_scalar_product
