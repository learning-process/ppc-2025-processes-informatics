

#include <cmath>
#include <cstddef>
#include <vector>
#include <mpi.h>
#include "zenin_a_topology_star/common/include/common.hpp"
#include "zenin_a_topology_star/seq/include/ops_seq.hpp"

namespace zenin_a_topology_star {

ZeninATopologyStarSEQ::ZeninATopologyStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninATopologyStarSEQ::ValidationImpl() {
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const auto& in = GetInput();
  size_t src = std::get<0>(in);
  size_t dst = std::get<1>(in);
  if (src >= static_cast<size_t>(world_size) || 
      dst >= static_cast<size_t>(world_size)) {
    return false;
  }
  return true;
}

bool ZeninATopologyStarSEQ::PreProcessingImpl() {
  return true;
}

bool ZeninATopologyStarSEQ::RunImpl() {
  int world_rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto& in = GetInput();
  const size_t src = std::get<0>(in);
  const size_t dst = std::get<1>(in);
  const auto& data = std::get<2>(in);

  auto& out = GetOutput();
  out.clear();

  const int center = 0;
  const int tag = 0;

  MPI_Comm star_comm;
  {
    const int nnodes = world_size;
    std::vector<int> index(nnodes);
    std::vector<int> edges;

    for (int i = 0; i < nnodes; ++i) {
      if (i == center) {
        for (int j = 0; j < nnodes; ++j) {
          if (j == center) continue;
          edges.push_back(j);
        }
      } else {
        edges.push_back(center);
      }
      index[i] = static_cast<int>(edges.size());
    }

    MPI_Graph_create(MPI_COMM_WORLD,
                     nnodes,
                     index.data(),
                     edges.data(),
                     0,
                     &star_comm);
  }

  
  int star_rank;
  MPI_Comm_rank(star_comm, &star_rank);  

  const int src_rank = static_cast<int>(src);
  const int dst_rank = static_cast<int>(dst);
  const int center_rank = center;

  
  if (src_rank == dst_rank) {
    if (world_rank == src_rank) {
      out = data;
    }
    MPI_Comm_free(&star_comm);
    return true;
  }

  
  if (src_rank == center_rank || dst_rank == center_rank) {
    if (world_rank == src_rank) {
      MPI_Send(data.data(), static_cast<int>(data.size()),
               MPI_DOUBLE, dst_rank, tag, star_comm);
    } else if (world_rank == dst_rank) {
      out.resize(data.size());
      MPI_Recv(out.data(), static_cast<int>(out.size()),
               MPI_DOUBLE, src_rank, tag, star_comm, MPI_STATUS_IGNORE);
    }

    MPI_Comm_free(&star_comm);
    return true;
  }

  
  if (world_rank == src_rank) {
    MPI_Send(data.data(), static_cast<int>(data.size()),
             MPI_DOUBLE, center_rank, tag, star_comm);
  } else if (world_rank == center_rank) {
    std::vector<double> buf(data.size());
    MPI_Recv(buf.data(), static_cast<int>(buf.size()),
             src_rank, tag, star_comm, MPI_STATUS_IGNORE);

    MPI_Send(buf.data(), static_cast<int>(buf.size()),
             dst_rank, tag, star_comm);
  } else if (world_rank == dst_rank) {
    out.resize(data.size());
    MPI_Recv(out.data(), static_cast<int>(out.size()),
             center_rank, tag, star_comm, MPI_STATUS_IGNORE);
  }

  MPI_Comm_free(&star_comm);
  return true;
}

bool ZeninATopologyStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_topology_star
