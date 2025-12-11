#include "shkrebko_m_hypercube/seq/include/ops_seq.hpp"

#include "shkrebko_m_hypercube/common/include/common.hpp"

namespace shkrebko_m_hypercube {

ShkrebkoMHypercubeSEQ::ShkrebkoMHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = HypercubeData();
}

bool ShkrebkoMHypercubeSEQ::ValidationImpl() {
  if (GetInput().size() < 2) {
    return false;
  }

  if (GetInput()[0] <= 0) {
    return false;
  }
  
  // В SEQ версии не проверяем destination, т.к. нет MPI_COMM_WORLD
  int destination = GetInput()[1];
  if (destination < 0) {
    return false;
  }

  return true;
}

bool ShkrebkoMHypercubeSEQ::PreProcessingImpl() {
  if (GetInput().size() >= 2) {
    GetOutput().value = GetInput()[0];
    GetOutput().destination = GetInput()[1];
  } else {
    GetOutput().value = GetInput()[0];
    GetOutput().destination = 0;
  }
  return true;
}

bool ShkrebkoMHypercubeSEQ::RunImpl() {
  GetOutput().path.push_back(0);
  GetOutput().finish = true;
  return true;
}

bool ShkrebkoMHypercubeSEQ::PostProcessingImpl() {
  return GetOutput().value > 0;
}

}  // namespace shkrebko_m_hypercube