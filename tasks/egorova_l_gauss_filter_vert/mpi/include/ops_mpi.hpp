// mpi/include/ops_mpi.hpp
#pragma once

#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"
#include "task/include/task.hpp"

namespace egorova_l_gauss_filter_vert {

class EgorovaLGaussFilterVertMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit EgorovaLGaussFilterVertMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace egorova_l_gauss_filter_vert
