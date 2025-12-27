#pragma once

#include "dolov_v_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dolov_v_torus_topology {

class DolovVTorusTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DolovVTorusTopologySEQ(InType in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void DefineGridDimensions(int total_procs, int &r, int &c);
  static int GetNextNode(int current, int target, int r, int c);

  InputData internal_input_;
  OutputData internal_output_;
};

}  // namespace dolov_v_torus_topology
