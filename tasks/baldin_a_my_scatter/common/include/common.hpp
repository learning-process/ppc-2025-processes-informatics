#pragma once

#include <string>
#include <tuple>
#include <vector>
#include <mpi.h>

#include "task/include/task.hpp"

namespace baldin_a_my_scatter {

using InType = std::tuple<const void*, int, MPI_Datatype, void *, int, MPI_Datatype, int, MPI_Comm>;
using OutType = void *;
using TestType = std::tuple<int, int, MPI_Datatype>; // count, root, type
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace baldin_a_my_scatter
