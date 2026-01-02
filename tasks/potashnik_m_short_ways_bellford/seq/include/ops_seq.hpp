#pragma once

#include "potashnik_m_short_ways_bellford/common/include/common.hpp"
#include "task/include/task.hpp"

namespace potashnik_m_short_ways_bellford {

class PotashnikMShortWaysBellfordSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PotashnikMShortWaysBellfordSEQ(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

inline void bellman_ford_algo_iteration_seq(const Graph& g, const std::vector<int>& dist, std::vector<int>& dist_next) {
  int n = g.n;
  dist_next = dist;
  for (int u = 0; u < n; u++) {
    if (dist[u] == 1e9) {
      continue;
    }
    iterate_through_vertex(g, u, dist, dist_next);
  }
}

inline void bellman_ford_algo_seq(const Graph& g, int source, std::vector<int>& dist) {
  const int INF = 1e9;
  int n = g.n;

  if (n == 0) {
    return;
  }
  if (source < 0 || source >= n) {
    return;
  }

  dist.assign(n, INF);
  dist[source] = 0;

  std::vector<int> dist_next;

  for (int i = 0; i < n - 1; i++) {
    bellman_ford_algo_iteration_seq(g, dist, dist_next);
    dist.swap(dist_next);
  }
}

}  // namespace potashnik_m_short_ways_bellford
