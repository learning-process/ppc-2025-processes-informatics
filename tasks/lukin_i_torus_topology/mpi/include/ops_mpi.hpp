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

  Direction GetDir(int sx, int sy, int dx, int dy, int cols, int rows);

  void Send(int &message_len, std::vector<int> &message, std::vector<int> &full_route, int &route_size, int dest,
            int rank) const;
  void Recieve(int &message_len, std::vector<int> &message, std::vector<int> &full_route, int &route_size,
               int source) const;
  bool HandleTrivial(int &message_len, std::vector<int> &message, int proc_count);
  void InitTopology(int &cols, int &rows, int proc_count) const;

  int start = -1;
  int end = -1;
};

}  // namespace lukin_i_torus_topology
