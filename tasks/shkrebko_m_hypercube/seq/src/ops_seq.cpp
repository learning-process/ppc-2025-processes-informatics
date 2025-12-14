#include "shkrebko_m_hypercube/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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
  for (int i = 0; i < 5000; i++) {
    volatile int dummy = 0;
    for (int j = 0; j < 5000; j++) {
      dummy += i * j;
    }
  }
  GetOutput().path.push_back(0);
  GetOutput().finish = true;
  return true;
}

bool ShkrebkoMHypercubeSEQ::PostProcessingImpl() {
  return GetOutput().value > 0;
}

}  // namespace shkrebko_m_hypercube
