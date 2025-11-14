#include "zavyalov_a_scalar_product/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <string>
#include <vector>

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
  std::vector<double> left;
  std::vector<double> right;

  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int vectorSize;

  if (rank == 0) {
    GetOutput() = 0.0;
    auto input = GetInput();
    if (!std::get<0>(input).empty()) {  // it does not compile in ubuntu without this line
      left = std::get<0>(input);
    }
    if (!std::get<1>(input).empty()) {  // it does not compile in ubuntu without this line
      right = std::get<1>(input);
    }
    vectorSize = left.size();
  }

  MPI_Bcast(&vectorSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  double *leftVecData = nullptr;
  double *rightVecData = nullptr;

  int blocksize = vectorSize / worldSize;
  int elementsLeft = vectorSize - worldSize * blocksize;

  if (rank == 0) {
    leftVecData = left.data();
    rightVecData = right.data();
    int procNumber;

    // send part of leftVecData to other processes
    int elementsProcessed = blocksize + (elementsLeft > 0);  // number of already processed elements
    for (procNumber = 1; procNumber < worldSize; ++procNumber) {
      MPI_Send(leftVecData + elementsProcessed, blocksize + (procNumber < elementsLeft), MPI_DOUBLE, procNumber, 0,
               MPI_COMM_WORLD);
      elementsProcessed += blocksize;
      if (procNumber < elementsLeft) {
        ++elementsProcessed;
      }
    }

    // send part of rightVecData to other processes
    elementsProcessed = blocksize + (elementsLeft > 0);

    for (procNumber = 1; procNumber < worldSize; ++procNumber) {
      MPI_Send(rightVecData + elementsProcessed, blocksize + (procNumber < elementsLeft), MPI_DOUBLE, procNumber, 1,
               MPI_COMM_WORLD);
      elementsProcessed += blocksize;
      if (procNumber < elementsLeft) {
        ++elementsProcessed;
      }
    }
    // check if process #0 have to calculate 1 more element
    if (elementsLeft > 0) {
      ++blocksize;
    }

  } else {
    if (rank < elementsLeft) {
      blocksize++;
    }

    leftVecData = new double[blocksize];
    rightVecData = new double[blocksize];

    int errorCode = MPI_Recv(leftVecData, blocksize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (errorCode != MPI_SUCCESS) {
      throw std::string("There was an error trying to receive leftVecData in process ") + std::to_string(rank);
    }
    errorCode = MPI_Recv(rightVecData, blocksize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (errorCode != MPI_SUCCESS) {
      throw std::string("There was an error trying to receive rightVecData in process ") + std::to_string(rank);
    }
  }
  double curRes = 0.0;
  if (leftVecData != nullptr && rightVecData != nullptr) {
    for (int i = 0; i < blocksize; i++) {
      curRes += leftVecData[i] * rightVecData[i];
    }
  }
  double globRes = 0.0;
  MPI_Allreduce(&curRes, &globRes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = globRes;
  if (rank != 0) {
    delete[] leftVecData;
    delete[] rightVecData;
  }

  return true;
}

bool ZavyalovAScalarProductMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_scalar_product
