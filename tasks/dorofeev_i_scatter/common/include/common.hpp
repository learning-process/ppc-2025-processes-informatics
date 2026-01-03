#pragma once

#include <mpi.h>

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dorofeev_i_scatter {

using InType = std::tuple<const void *, int, MPI_Datatype, void *, int, MPI_Datatype, int, MPI_Comm>;
using OutType = void *;
using TestType = std::tuple<int, int, MPI_Datatype>;  // count, root, type
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dorofeev_i_scatter
