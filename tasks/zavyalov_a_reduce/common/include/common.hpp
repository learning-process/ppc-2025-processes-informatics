#pragma once

#include <tuple>
#include <vector>
#include <mpi.h>

#include "task/include/task.hpp"

namespace zavyalov_a_reduce {
// InType: операция, тип данных, число элементов в передаваемом массиве, указатель на память где хранится массив, номер процесса-получателя
using InType = std::tuple<MPI_Op, MPI_Datatype, size_t, void*, int>; // void* instead of vector because input type can differ
using OutType = void*; // result of mpi_reduce. void* instead of vector because input type can differ
using TestType = std::tuple<MPI_Op, MPI_Datatype, size_t, int>;  // operation type, vector elements type, size of vectors, receiver process rank
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zavyalov_a_reduce
