#pragma once

#include <tuple>
#include <vector>
#include <mpi.h>

#include "task/include/task.hpp"

namespace zavyalov_a_reduce {

using InType = std::tuple<MPI_Op, MPI_Datatype, unsigned int, void*>; // void* instead of vector because input type can differ
using OutType = void*; // result of mpi_reduce. void* instead of vector because input type can differ
using TestType = std::tuple<MPI_Op, MPI_Datatype, unsigned int>;  // operation type, vector elements type, size of vectors
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zavyalov_a_reduce
