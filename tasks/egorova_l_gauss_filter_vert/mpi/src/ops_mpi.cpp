#include "egorova_l_find_max_val_col_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace egorova_l_gauss_filter_vert {

EgorovaLGaussFilterVertMPI::EgorovaLGaussFilterVertMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLGaussFilterVertMPI::ValidationImpl() {
  const auto &in = GetInput();
  return in.rows > 0 && in.cols > 0 && in.channels > 0;
}

bool EgorovaLGaussFilterVertMPI::PreProcessingImpl() { return true; }

void EgorovaLGaussFilterVertMPI::ApplyFilter(const std::vector<uint8_t> &local_in, 
                                            std::vector<uint8_t> &local_out,
                                            int rows, int local_cols, int total_lc, int ch) {
  const std::array<std::array<float, 3>, 3> ker = {{
      {0.0625F, 0.125F, 0.0625F}, {0.125F, 0.25F, 0.125F}, {0.0625F, 0.125F, 0.0625F}
  }};

  for (int yy = 0; yy < rows; ++yy) {
    for (int xx = 0; xx < local_cols; ++xx) {
      for (int cc = 0; cc < ch; ++cc) {
        float sum = 0.0F;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int py = std::clamp(yy + ky, 0, rows - 1);
            int px = xx + kx + 1;
            sum += static_cast<float>(local_in[((py * total_lc) + px) * ch + cc]) * ker[static_cast<std::size_t>(ky + 1)][static_cast<std::size_t>(kx + 1)];
          }
        }
        local_out[((yy * local_cols) + xx) * ch + cc] = static_cast<uint8_t>(sum);
      }
    }
  }
}

bool EgorovaLGaussFilterVertMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &full_img = GetInput();
  int rr = full_img.rows;
  int cc = full_img.cols;
  int ch = full_img.channels;

  int qq = cc / size;
  int rem = cc % size;
  int local_cols = qq + (rank < rem ? 1 : 0);
  int offset = (rank * qq) + std::min(rank, rem);

  int left_h = (rank > 0) ? 1 : 0;
  int right_h = (rank < size - 1) ? 1 : 0;
  int total_lc = local_cols + left_h + right_h;

  std::vector<uint8_t> local_in(static_cast<std::size_t>(rr) * total_lc * ch);
  std::vector<uint8_t> local_out(static_cast<std::size_t>(rr) * local_cols * ch);

  for (int ii = 0; ii < rr; ++ii) {
    std::copy(full_img.data.begin() + static_cast<std::ptrdiff_t>(((ii * cc) + offset - left_h) * ch),
              full_img.data.begin() + static_cast<std::ptrdiff_t>(((ii * cc) + offset + local_cols + right_h) * ch),
              local_in.begin() + static_cast<std::ptrdiff_t>((ii * total_lc * ch)));
  }

  ApplyFilter(local_in, local_out, rr, local_cols, total_lc, ch);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  for (int ii = 0; ii < size; ++ii) {
    counts[ii] = (qq + (ii < rem ? 1 : 0)) * ch;
    displs[ii] = (ii == 0) ? 0 : displs[ii - 1] + counts[ii - 1];
  }

  auto &output = GetOutput();
  if (rank == 0) {
    output.rows = rr;
    output.cols = cc;
    output.channels = ch;
    output.data.resize(static_cast<std::size_t>(rr) * cc * ch);
  }

  for (int ii = 0; ii < rr; ++ii) {
    MPI_Gatherv(local_out.data() + static_cast<std::ptrdiff_t>(ii * local_cols * ch), local_cols * ch, MPI_BYTE,
                output.data.data() + static_cast<std::ptrdiff_t>(ii * cc * ch), counts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);
  }

  return true;
}

bool EgorovaLGaussFilterVertMPI::PostProcessingImpl() { return true; }

}  // namespace egorova_l_gauss_filter_vert