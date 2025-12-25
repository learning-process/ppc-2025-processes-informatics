#include "dolov_v_torus_topology/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dolov_v_torus_topology {

DolovVTorusTopologySEQ::DolovVTorusTopologySEQ(const InType &in) : internal_input_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool DolovVTorusTopologySEQ::ValidationImpl() {
  // Проверяем, что ранги в пределах созданной сети
  return internal_input_.sender_rank >= 0 && internal_input_.receiver_rank >= 0 &&
         internal_input_.sender_rank < internal_input_.total_procs &&
         internal_input_.receiver_rank < internal_input_.total_procs && !internal_input_.message.empty();
}

bool DolovVTorusTopologySEQ::PreProcessingImpl() {
  internal_output_.route.clear();
  internal_output_.received_message.clear();
  return true;
}

bool DolovVTorusTopologySEQ::RunImpl() {
  internal_output_.received_message = internal_input_.message;
  int current_node = internal_input_.sender_rank;
  internal_output_.route = {current_node};

  if (internal_input_.sender_rank == internal_input_.receiver_rank) {
    return true;
  }

  int t_procs = internal_input_.total_procs;
  int r = 0;
  int c = 0;

  // Логика определения размеров должна быть идентична MPI
  r = static_cast<int>(std::sqrt(t_procs));
  while (t_procs % r != 0) {
    r--;
  }
  c = t_procs / r;

  while (current_node != internal_input_.receiver_rank) {
    int curr_x = current_node % c;
    int curr_y = current_node / c;
    int tar_x = internal_input_.receiver_rank % c;
    int tar_y = internal_input_.receiver_rank / c;

    int dx = tar_x - curr_x;
    int dy = tar_y - curr_y;

    // Кратчайший путь на торе (учитываем "склейку" границ)
    if (std::abs(dx) > c / 2) {
      dx = (dx > 0) ? dx - c : dx + c;
    }
    if (std::abs(dy) > r / 2) {
      dy = (dy > 0) ? dy - r : dy + r;
    }

    // Сначала двигаемся по X, потом по Y
    if (dx != 0) {
      curr_x = (dx > 0) ? (curr_x + 1) % c : (curr_x - 1 + c) % c;
    } else if (dy != 0) {
      curr_y = (dy > 0) ? (curr_y + 1) % r : (curr_y - 1 + r) % r;
    }

    current_node = curr_y * c + curr_x;
    internal_output_.route.push_back(current_node);
  }

  return true;
}

bool DolovVTorusTopologySEQ::PostProcessingImpl() {
  GetOutput() = internal_output_;
  return true;
}

}  // namespace dolov_v_torus_topology
