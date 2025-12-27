#include "dolov_v_torus_topology/seq/include/ops_seq.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include "dolov_v_torus_topology/common/include/common.hpp"

namespace dolov_v_torus_topology {

DolovVTorusTopologySEQ::DolovVTorusTopologySEQ(InType in) : internal_input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool DolovVTorusTopologySEQ::ValidationImpl() {
  return internal_input_.sender_rank >= 0 && internal_input_.receiver_rank >= 0 &&
         internal_input_.sender_rank < internal_input_.total_procs &&
         internal_input_.receiver_rank < internal_input_.total_procs && !internal_input_.message.empty();
}

bool DolovVTorusTopologySEQ::PreProcessingImpl() {
  internal_output_.route.clear();
  internal_output_.received_message.clear();
  return true;
}

void DolovVTorusTopologySEQ::DefineGridDimensions(int total_procs, int &r, int &c) {
  r = static_cast<int>(std::sqrt(total_procs));
  while (total_procs % r != 0) {
    r--;
  }
  c = total_procs / r;
}

int DolovVTorusTopologySEQ::GetNextNode(int current, int target, int r, int c) {
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
    curr_x = (dx > 0) ? (curr_x + 1) % c : (curr_x - 1 + c) % c;
  } else if (dy != 0) {
    curr_y = (dy > 0) ? (curr_y + 1) % r : (curr_y - 1 + r) % r;
  }

  return (curr_y * c) + curr_x;
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
  DefineGridDimensions(t_procs, r, c);

  while (current_node != internal_input_.receiver_rank) {
    current_node = GetNextNode(current_node, internal_input_.receiver_rank, r, c);
    internal_output_.route.push_back(current_node);
  }

  return true;
}

bool DolovVTorusTopologySEQ::PostProcessingImpl() {
  GetOutput() = internal_output_;
  return true;
}

}  // namespace dolov_v_torus_topology
