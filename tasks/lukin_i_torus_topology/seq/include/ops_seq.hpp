#pragma once

#include "lukin_i_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lukin_i_torus_topology {

class LukinIThorTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LukinIThorTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool HandleTrivial(int &message_len, std::vector<int> &message, int proc_count);
  void InitTopology(int &cols, int &rows, int proc_count) const;

  int start = -1;
  int end = -1;

  const std::vector<int> dummy_route = {2, 0, 2, 6};  // MPI скрывает маршрутизацию
};
}  // namespace lukin_i_torus_topology
