#include "yurkin_g_ruler/seq/include/ops_seq.hpp"

#include "yurkin_g_ruler/common/include/common.hpp"

namespace yurkin_g_ruler {

YurkinGRulerSEQ::YurkinGRulerSEQ(const InType &in) {
  SetTypeOfTask(YurkinGRulerSEQ::GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGRulerSEQ::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinGRulerSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinGRulerSEQ::RunImpl() {
  volatile int acc = 0;
  const int n = GetInput();

  for (int i = 0; i < n; ++i) {
    acc += i % 7;
  }

  GetOutput() = n;
  return true;
}

bool YurkinGRulerSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_g_ruler
