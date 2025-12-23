#pragma once

#include "task/include/task.hpp"
#include "zenin_a_gauss_filter/common/include/common.hpp"

namespace zenin_a_gauss_filter {

class ZeninAGaussFilterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZeninAGaussFilterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zenin_a_gauss_filter
