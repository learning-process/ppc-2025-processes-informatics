#pragma once

#include <mpi.h>

#include "potashnik_m_short_ways_bellford/common/include/common.hpp"
#include "task/include/task.hpp"

namespace potashnik_m_short_ways_bellford {

class PotashnikMShortWaysBellfordMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PotashnikMShortWaysBellfordMPI(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

inline void bellman_ford_algo_iteration_mpi(const Graph& g, const std::vector<int>& dist, std::vector<int>& dist_next,
                                            int start, int end) {
  dist_next = dist;
  for (int u = start; u < end; u++) {
    if (dist[u] == 1e9) {
      continue;
    }
    iterate_through_vertex(g, u, dist, dist_next);
  }
}

inline void bellman_ford_algo_mpi(const Graph& g, int source, std::vector<int>& dist) {
  const int INF = 1e9;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = g.n;

  dist.assign(n, INF);
  if (rank == 0) {
    dist[source] = 0;
  }

  MPI_Bcast(dist.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> dist_next;

  int start = rank * n / size;
  int end = (rank + 1) * n / size;

  for (int i = 0; i < n - 1; i++) {
    bellman_ford_algo_iteration_mpi(g, dist, dist_next, start, end);
    MPI_Allreduce(dist_next.data(), dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
}

}  // namespace potashnik_m_short_ways_bellford
