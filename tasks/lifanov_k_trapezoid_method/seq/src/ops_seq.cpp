#include "lifanov_k_trapezoid_method/seq/include/ops_seq.hpp"

#include <vector>

#include "lifanov_k_trapezoid_method/common/include/common.hpp"

namespace lifanov_k_trapezoid_method {

LifanovKTrapezoidMethodSEQ::LifanovKTrapezoidMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LifanovKTrapezoidMethodSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool LifanovKTrapezoidMethodSEQ::PreProcessingImpl() {
  return true;
}

bool LifanovKTrapezoidMethodSEQ::RunImpl() {
  std::vector<int> current_values = GetInput();
  GetOutput().resize(1);
  int global_sum = 0;
  for (int current_value : current_values) {
    global_sum += current_value;
  }
  GetOutput()[0] = global_sum;

  return true;
}

bool LifanovKTrapezoidMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_trapezoid_method
