#include "dolov_v_torus_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace dolov_v_torus_topology {

DolovVTorusTopologyMPI::DolovVTorusTopologyMPI(const InType &in) : input_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool DolovVTorusTopologyMPI::ValidationImpl() {
  int proc_count = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  return input_.sender_rank >= 0 && input_.sender_rank < proc_count && input_.receiver_rank >= 0 &&
         input_.receiver_rank < proc_count && input_.total_procs == proc_count;
}

bool DolovVTorusTopologyMPI::PreProcessingImpl() {
  output_.route.clear();
  output_.received_message.clear();
  return true;
}

bool DolovVTorusTopologyMPI::RunImpl() {
  int rank = 0;
  int proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  int rows = 0;
  int cols = 0;
  DefineGridDimensions(proc_count, rows, cols);

  int msg_size = (rank == input_.sender_rank) ? static_cast<int>(input_.message.size()) : 0;
  MPI_Bcast(&msg_size, 1, MPI_INT, input_.sender_rank, MPI_COMM_WORLD);

  std::vector<int> current_msg;
  if (rank == input_.sender_rank) {
    current_msg = input_.message;
  } else {
    current_msg.resize(static_cast<size_t>(msg_size));
  }

  int current_node = input_.sender_rank;
  std::vector<int> path = {current_node};

  // Итеративная передача данных по узлам
  while (current_node != input_.receiver_rank) {
    MoveSide next_step = FindShortestPathStep(current_node, input_.receiver_rank, rows, cols);
    int next_node = GetTargetNeighbor(current_node, next_step, rows, cols);

    if (rank == current_node) {
      // Отправляем сообщение и текущий накопленный путь
      MPI_Send(current_msg.data(), msg_size, MPI_INT, next_node, ProtocolTags::kDataTransfer, MPI_COMM_WORLD);
      int p_size = static_cast<int>(path.size());
      MPI_Send(&p_size, 1, MPI_INT, next_node, ProtocolTags::kRouteSync, MPI_COMM_WORLD);
      MPI_Send(path.data(), p_size, MPI_INT, next_node, ProtocolTags::kRouteSync, MPI_COMM_WORLD);
    } else if (rank == next_node) {
      // Принимаем данные
      MPI_Recv(current_msg.data(), msg_size, MPI_INT, current_node, ProtocolTags::kDataTransfer, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      int p_size = 0;
      MPI_Recv(&p_size, 1, MPI_INT, current_node, ProtocolTags::kRouteSync, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      path.resize(static_cast<size_t>(p_size));
      MPI_Recv(path.data(), p_size, MPI_INT, current_node, ProtocolTags::kRouteSync, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      path.push_back(next_node);
    }

    // Все процессы синхронно обновляют текущий узел-владелец
    int prev_owner = current_node;
    current_node = next_node;
    MPI_Bcast(&current_node, 1, MPI_INT, prev_owner, MPI_COMM_WORLD);
  }

  // Финальный сбор результатов (чтобы на каждом ранге был верный Output)
  if (rank == input_.receiver_rank) {
    output_.received_message = std::move(current_msg);
    output_.route = std::move(path);
  }

  // Рассылаем финальный путь всем для прохождения тестов
  int final_path_size = static_cast<int>(output_.route.size());
  MPI_Bcast(&final_path_size, 1, MPI_INT, input_.receiver_rank, MPI_COMM_WORLD);
  if (rank != input_.receiver_rank) {
    output_.route.resize(static_cast<size_t>(final_path_size));
  }
  MPI_Bcast(output_.route.data(), final_path_size, MPI_INT, input_.receiver_rank, MPI_COMM_WORLD);

  // Рассылаем полученное сообщение всем
  if (rank != input_.receiver_rank) {
    output_.received_message.resize(static_cast<size_t>(msg_size));
  }
  MPI_Bcast(output_.received_message.data(), msg_size, MPI_INT, input_.receiver_rank, MPI_COMM_WORLD);

  return true;
}

void DolovVTorusTopologyMPI::DefineGridDimensions(int total_procs, int &r, int &c) {
  r = static_cast<int>(std::sqrt(total_procs));
  while (total_procs % r != 0) {
    r--;
  }
  c = total_procs / r;
}

DolovVTorusTopologyMPI::MoveSide DolovVTorusTopologyMPI::FindShortestPathStep(int current, int target, int r, int c) {
  int curr_x = current % c;
  int curr_y = current / c;
  int tar_x = target % c;
  int tar_y = target / c;

  int dx = tar_x - curr_x;
  int dy = tar_y - curr_y;

  if (std::abs(dx) > c / 2) {
    dx = (dx > 0) ? dx - c : dx + c;
  }
  if (std::abs(dy) > r / 2) {
    dy = (dy > 0) ? dy - r : dy + r;
  }

  if (dx != 0) {
    return (dx > 0) ? MoveSide::kEast : MoveSide::kWest;
  }
  if (dy != 0) {
    return (dy > 0) ? MoveSide::kSouth : MoveSide::kNorth;
  }
  return MoveSide::kStay;
}

int DolovVTorusTopologyMPI::GetTargetNeighbor(int current, MoveSide side, int r, int c) {
  int x = current % c;
  int y = current / c;
  switch (side) {
    case MoveSide::kNorth:
      y = (y - 1 + r) % r;
      break;
    case MoveSide::kSouth:
      y = (y + 1) % r;
      break;
    case MoveSide::kWest:
      x = (x - 1 + c) % c;
      break;
    case MoveSide::kEast:
      x = (x + 1) % c;
      break;
    case MoveSide::kStay:
      break;
  }
  return y * c + x;
}

bool DolovVTorusTopologyMPI::PostProcessingImpl() {
  GetOutput() = output_;
  return true;
}

}  // namespace dolov_v_torus_topology
