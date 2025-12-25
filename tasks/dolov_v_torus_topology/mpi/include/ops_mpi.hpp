#pragma once

#include "dolov_v_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_torus_topology {

class DolovVTorusTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DolovVTorusTopologyMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dolov_v_torus_topology
