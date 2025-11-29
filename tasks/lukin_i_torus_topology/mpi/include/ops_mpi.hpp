#pragma once

#include "lukin_i_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lukin_i_torus_topology {

class LukinIThorTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LukinIThorTopologyMPI(const InType &in);

 private:
  enum class Direction { UP, DOWN, LEFT, RIGHT, NONE };

  enum class Tags { ROUTESIZE, ROUTE, MESSAGE, MLEN };

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  Direction get_dir(const int sx, const int sy, const int dx, const int dy, const int cols, const int rows);

  int start = -1;
  int end = -1;
};

}  // namespace lukin_i_torus_topology
