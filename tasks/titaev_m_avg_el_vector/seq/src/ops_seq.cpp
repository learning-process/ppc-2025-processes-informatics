#include "titaev_m_avg_el_vector/seq/include/ops_seq.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

namespace titaev_m_avg_el_vector {

TitaevMAvgElVectorSEQ::TitaevMAvgElVectorSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TitaevMAvgElVectorSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool TitaevMAvgElVectorSEQ::PreProcessingImpl() {
  return true;
}

bool TitaevMAvgElVectorSEQ::RunImpl() {
  const auto &input_vec = GetInput();

  if (input_vec.empty()) {
    GetOutput() = 0.0;
    return true;
  }

  int64_t sum = std::accumulate(input_vec.begin(), input_vec.end(), 0LL);

  GetOutput() = static_cast<double>(sum) / input_vec.size();

  return true;
}

bool TitaevMAvgElVectorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_avg_el_vector
