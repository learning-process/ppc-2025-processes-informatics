#include "lukin_i_torus_topology/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"

namespace lukin_i_torus_topology {

LukinIThorTopologySEQ::LukinIThorTopologySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::vector<int> message = std::get<2>(GetInput());
  GetOutput() = std::make_tuple(dummy_route, message);
}

bool LukinIThorTopologySEQ::ValidationImpl() {
  return true;
}

bool LukinIThorTopologySEQ::PreProcessingImpl() {
  return true;
}

bool LukinIThorTopologySEQ::RunImpl() {
  return true;
}

bool LukinIThorTopologySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_torus_topology
