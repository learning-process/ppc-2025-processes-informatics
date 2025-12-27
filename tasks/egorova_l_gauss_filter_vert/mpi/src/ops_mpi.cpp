#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"

namespace egorova_l_gauss_filter_vert {

namespace {
void CalculateCountsAndDispls(int size, int qq, int rem, int rows, int ch, std::vector<int>& counts,
                             std::vector<int>& displs) {
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    int cur_cols = qq + (i < rem ? 1 : 0);
    counts.at(static_cast<size_t>(i)) = cur_cols * rows * ch;
    displs.at(static_cast<size_t>(i)) = offset;
    offset += counts.at(static_cast<size_t>(i));
  }
}
}  // namespace

EgorovaLGaussFilterVertMPI::EgorovaLGaussFilterVertMPI(const InType& in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLGaussFilterVertMPI::ValidationImpl() {
  const auto& in = GetInput();
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::array<int, 3> params{};
  if (rank == 0) {
    params[0] = in.rows;
    params[1] = in.cols;
    params[2] = in.channels;
  }
  MPI_Bcast(params.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  return (params[0] > 0 && params[1] > 0 && params[2] > 0);
}

bool EgorovaLGaussFilterVertMPI::PreProcessingImpl() { return true; }

void EgorovaLGaussFilterVertMPI::ApplyFilter(const std::vector<uint8_t>& local_in, std::vector<uint8_t>& local_out,
                                             int rows, int local_cols, int total_lc, int ch) {
  const std::array<float, 9> ker = {0.0625F, 0.125F, 0.0625F, 0.125F, 0.25F, 0.125F, 0.0625F, 0.125F, 0.0625F};
  for (int yy = 0; yy < rows; ++yy) {
    for (int xx = 0; xx < local_cols; ++xx) {
      for (int cc = 0; cc < ch; ++cc) {
        float sum = 0.0F;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int py = std::clamp(yy + ky, 0, rows - 1);
            int px = std::clamp(xx + kx + 1, 0, total_lc - 1);
            size_t in_idx = ((static_cast<size_t>(py) * total_lc + px) * ch) + cc;
            size_t k_idx = (static_cast<size_t>(ky + 1) * 3) + (kx + 1);
            sum += (static_cast<float>(local_in[in_idx]) * ker.at(k_idx));
          }
        }
        size_t out_idx = ((static_cast<size_t>(yy) * local_cols + xx) * ch) + cc;
        local_out[out_idx] = static_cast<uint8_t>(std::clamp(std::round(sum), 0.0F, 255.0F));
      }
    }
  }
}

void EgorovaLGaussFilterVertMPI::FillLocalWithHalo(const std::vector<uint8_t>& local_data,
                                                   std::vector<uint8_t>& local_with_halo,
                                                   int rows, int local_cols, int total_lc, int left_h, int ch) {
  for (int row = 0; row < rows; ++row) {
    size_t src_off = static_cast<size_t>(row) * local_cols * ch;
    size_t dst_off = (static_cast<size_t>(row) * total_lc + left_h) * ch;
    std::copy(local_data.begin() + static_cast<std::ptrdiff_t>(src_off),
              local_data.begin() + static_cast<std::ptrdiff_t>(src_off + (static_cast<size_t>(local_cols) * ch)),
              local_with_halo.begin() + static_cast<std::ptrdiff_t>(dst_off));
  }
}

void EgorovaLGaussFilterVertMPI::ExchangeHalo(std::vector<uint8_t>& local_with_halo, int target, int rows,
                                              int total_lc, int ch, size_t send_col, size_t recv_col) {
  if (target == MPI_PROC_NULL) {
    return;
  }
  std::vector<uint8_t> s_buf(static_cast<size_t>(rows) * ch);
  std::vector<uint8_t> r_buf(static_cast<size_t>(rows) * ch);
  for (int i = 0; i < rows; ++i) {
    size_t s_idx = (static_cast<size_t>(i) * total_lc + send_col) * ch;
    std::copy(local_with_halo.begin() + static_cast<std::ptrdiff_t>(s_idx),
              local_with_halo.begin() + static_cast<std::ptrdiff_t>(s_idx + ch),
              s_buf.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(i) * ch));
  }
  MPI_Sendrecv(s_buf.data(), static_cast<int>(s_buf.size()), MPI_BYTE, target, 0, r_buf.data(),
               static_cast<int>(r_buf.size()), MPI_BYTE, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (int i = 0; i < rows; ++i) {
    size_t d_idx = (static_cast<size_t>(i) * total_lc + recv_col) * ch;
    std::copy(r_buf.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(i) * ch),
              r_buf.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(i + 1) * ch),
              local_with_halo.begin() + static_cast<std::ptrdiff_t>(d_idx));
  }
}

bool EgorovaLGaussFilterVertMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rr = 0;
  int cc = 0;
  int ch = 0;
  if (rank == 0) {
    rr = GetInput().rows;
    cc = GetInput().cols;
    ch = GetInput().channels;
  }
  MPI_Bcast(&rr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ch, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int qq = cc / size;
  int rem = cc % size;
  int local_cols = qq + (rank < rem ? 1 : 0);

  std::vector<int> counts(static_cast<size_t>(size));
  std::vector<int> displs(static_cast<size_t>(size));
  CalculateCountsAndDispls(size, qq, rem, rr, ch, counts, displs);

  std::vector<uint8_t> local_data(static_cast<size_t>(rr) * local_cols * ch);
  void* send_ptr = (rank == 0) ? GetInput().data.data() : nullptr;
  MPI_Scatterv(send_ptr, counts.data(), displs.data(), MPI_BYTE, local_data.data(),
               counts[static_cast<size_t>(rank)], MPI_BYTE, 0, MPI_COMM_WORLD);

  int left_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  int right_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
  int left_h = (left_rank != MPI_PROC_NULL) ? 1 : 0;
  int right_h = (right_rank != MPI_PROC_NULL) ? 1 : 0;
  int total_lc = local_cols + left_h + right_h;

  std::vector<uint8_t> local_with_halo(static_cast<size_t>(rr) * total_lc * ch, 0);
  FillLocalWithHalo(local_data, local_with_halo, rr, local_cols, total_lc, left_h, ch);
  
  ExchangeHalo(local_with_halo, right_rank, rr, total_lc, ch, static_cast<size_t>(left_h + local_cols - 1), 
               static_cast<size_t>(left_h + local_cols));
  ExchangeHalo(local_with_halo, left_rank, rr, total_lc, ch, static_cast<size_t>(left_h), 0);

  std::vector<uint8_t> local_out(static_cast<size_t>(rr) * local_cols * ch);
  ApplyFilter(local_with_halo, local_out, rr, local_cols, total_lc, ch);

  if (rank == 0) {
    GetOutput().rows = rr;
    GetOutput().cols = cc;
    GetOutput().channels = ch;
    GetOutput().data.resize(static_cast<size_t>(rr) * cc * ch);
  }

  void* recv_ptr = (rank == 0) ? GetOutput().data.data() : nullptr;
  MPI_Gatherv(local_out.data(), static_cast<int>(local_out.size()), MPI_BYTE,
              recv_ptr, counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

  return true;
}

bool EgorovaLGaussFilterVertMPI::PostProcessingImpl() { return true; }

}  // namespace egorova_l_gauss_filter_vert