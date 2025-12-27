#include "yurkin_g_ruler/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "yurkin_g_ruler/common/include/common.hpp"

namespace yurkin_g_ruler {

namespace {

struct Participation {
  bool will_recv;
  int recv_source;
  bool will_send;
  int send_target;
  int send_val;
};

Participation ComputeParticipation(int rank, int src, int dst, int low, int high, int payload) {
  if (rank == src) {
    int next = rank + ((dst > src) ? +1 : -1);
    if (next >= low && next <= high) {
      return Participation{.will_recv = false,
                           .recv_source = MPI_PROC_NULL,
                           .will_send = true,
                           .send_target = next,
                           .send_val = payload};
    }
    return Participation{.will_recv = false,
                         .recv_source = MPI_PROC_NULL,
                         .will_send = false,
                         .send_target = MPI_PROC_NULL,
                         .send_val = 0};
  }
  int direction = (dst > src) ? +1 : -1;
  Participation p{.will_recv = false,
                  .recv_source = MPI_PROC_NULL,
                  .will_send = false,
                  .send_target = MPI_PROC_NULL,
                  .send_val = 0};
  int prev = rank - direction;
  if (prev >= low && prev <= high) {
    p.will_recv = true;
    p.recv_source = prev;
  }
  if (rank != dst) {
    int next = rank + direction;
    if (next >= low && next <= high) {
      p.will_send = true;
      p.send_target = next;
    }
  }
  return p;
}

void PerformReceiveIfNeeded(MPI_Comm comm, int rank, int dst, const Participation &p, int &out_val) {
  if (!p.will_recv) {
    return;
  }
  int recv_val = 0;
  MPI_Recv(&recv_val, 1, MPI_INT, p.recv_source, 0, comm, MPI_STATUS_IGNORE);
  if (rank == dst) {
    out_val = recv_val;
  }
}

void PerformSendIfNeeded(MPI_Comm comm, const Participation &p, int send_val) {
  if (!p.will_send || p.send_target == MPI_PROC_NULL) {
    return;
  }
  MPI_Send(&send_val, 1, MPI_INT, p.send_target, 0, comm);
}

}  // namespace

YurkinGRulerMPI::YurkinGRulerMPI(const InType &in) {
  SetTypeOfTask(YurkinGRulerMPI::GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinGRulerMPI::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinGRulerMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinGRulerMPI::RunImpl() {
  MPI_Comm topo_comm = MPI_COMM_NULL;
  MPI_Comm_dup(MPI_COMM_WORLD, &topo_comm);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(topo_comm, &rank);
  MPI_Comm_size(topo_comm, &size);

  if (size <= 0) {
    if (topo_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&topo_comm);
    }
    return false;
  }

  const int input_value = GetInput();
  const int src = input_value % size;
  const int dst = (input_value / size) % size;
  const int payload = input_value;

  if (src == dst) {
    if (rank == dst) {
      GetOutput() = payload;
    }
    MPI_Barrier(topo_comm);
    if (topo_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&topo_comm);
    }
    return true;
  }

  const int low = std::min(src, dst);
  const int high = std::max(src, dst);

  if (rank < low || rank > high) {
    MPI_Barrier(topo_comm);
    if (topo_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&topo_comm);
    }
    return true;
  }

  Participation p = ComputeParticipation(rank, src, dst, low, high, payload);

  int recv_val = 0;
  if (p.will_recv) {
    MPI_Recv(&recv_val, 1, MPI_INT, p.recv_source, 0, topo_comm, MPI_STATUS_IGNORE);
    if (rank == dst) {
      GetOutput() = recv_val;
    } else {
      PerformSendIfNeeded(topo_comm, p, recv_val);
    }
  } else {
    if (p.will_send && p.send_target != MPI_PROC_NULL) {
      PerformSendIfNeeded(topo_comm, p, p.send_val);
    }
  }

  MPI_Barrier(topo_comm);
  if (topo_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&topo_comm);
  }
  return true;
}

bool YurkinGRulerMPI::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_g_ruler
