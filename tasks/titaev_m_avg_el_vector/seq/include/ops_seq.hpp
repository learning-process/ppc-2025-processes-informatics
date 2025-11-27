#pragma once

#include "titaev_m_avg_el_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace titaev_m_avg_el_vector {

class TitaevMAvgElVectorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TitaevMAvgElVectorSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace titaev_m_avg_el_vector
