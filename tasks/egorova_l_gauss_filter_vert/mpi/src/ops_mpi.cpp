#include "egorova_l_gauss_filter_vert/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "egorova_l_gauss_filter_vert/common/include/common.hpp"

namespace egorova_l_gauss_filter_vert {

EgorovaLGaussFilterVertMPI::EgorovaLGaussFilterVertMPI(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool EgorovaLGaussFilterVertMPI::ValidationImpl() {
  const auto &in = GetInput();
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

bool EgorovaLGaussFilterVertMPI::PreProcessingImpl() {
  return true;
}

void EgorovaLGaussFilterVertMPI::ApplyFilter(const std::vector<uint8_t> &local_in, std::vector<uint8_t> &local_out,
                                             int rows, int local_cols, int total_lc, int ch) {
  const std::array<float, 9> ker = {0.0625F, 0.125F, 0.0625F, 0.125F, 0.25F, 0.125F, 0.0625F, 0.125F, 0.0625F};
  const float *ker_ptr = ker.data();

  for (int yy = 0; yy < rows; ++yy) {
    for (int xx = 0; xx < local_cols; ++xx) {
      for (int cc = 0; cc < ch; ++cc) {
        float sum = 0.0F;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            int py = std::clamp(yy + ky, 0, rows - 1);
            int px = xx + kx + 1;

            std::size_t in_idx =
                ((((static_cast<std::size_t>(py) * static_cast<std::size_t>(total_lc)) + static_cast<std::size_t>(px)) *
                  static_cast<std::size_t>(ch)) +
                 static_cast<std::size_t>(cc));

            std::size_t k_idx = (static_cast<std::size_t>(ky + 1) * 3) + static_cast<std::size_t>(kx + 1);
            sum += (static_cast<float>(local_in[in_idx]) * ker_ptr[k_idx]);
          }
        }
        std::size_t out_idx =
            ((((static_cast<std::size_t>(yy) * static_cast<std::size_t>(local_cols)) + static_cast<std::size_t>(xx)) *
              static_cast<std::size_t>(ch)) +
             static_cast<std::size_t>(cc));

        local_out[out_idx] = static_cast<uint8_t>(sum);
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
  int rr = 0;
  int cc = 0;
  int ch = 0;

  if (rank == 0) {
    rr = full_img.rows;
    cc = full_img.cols;
    ch = full_img.channels;
  }

  MPI_Bcast(&rr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ch, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int qq = (cc / size);
  int rem = (cc % size);
  int local_cols = (qq + (rank < rem ? 1 : 0));
  int offset = ((rank * qq) + std::min(rank, rem));

  int left_h = ((rank > 0) ? 1 : 0);
  int right_h = ((rank < size - 1) ? 1 : 0);
  int total_lc = (local_cols + left_h + right_h);

  std::vector<uint8_t> local_in(static_cast<std::size_t>(rr) * total_lc * ch);
  std::vector<uint8_t> local_out(static_cast<std::size_t>(rr) * local_cols * ch);

  std::vector<uint8_t> shared_data;
  if (rank == 0) {
    shared_data = full_img.data;
  } else {
    shared_data.resize(static_cast<std::size_t>(rr) * cc * ch);
  }
  MPI_Bcast(shared_data.data(), static_cast<int>(shared_data.size()), MPI_BYTE, 0, MPI_COMM_WORLD);

  for (int ii = 0; ii < rr; ++ii) {
    std::size_t src_start = (static_cast<std::size_t>(ii) * cc + (offset - left_h)) * ch;
    std::size_t dest_start = (static_cast<std::size_t>(ii) * total_lc) * ch;
    std::size_t count = static_cast<std::size_t>(total_lc) * ch;

    std::copy(std::next(shared_data.begin(), static_cast<std::ptrdiff_t>(src_start)),
              std::next(shared_data.begin(), static_cast<std::ptrdiff_t>(src_start + count)),
              std::next(local_in.begin(), static_cast<std::ptrdiff_t>(dest_start)));
  }

  ApplyFilter(local_in, local_out, rr, local_cols, total_lc, ch);

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  for (int ii = 0; ii < size; ++ii) {
    counts[ii] = (qq + (ii < rem ? 1 : 0)) * ch;
    displs[ii] = (ii == 0) ? 0 : displs[ii - 1] + counts[ii - 1];
  }

  auto &output = GetOutput();
  output.rows = rr;
  output.cols = cc;
  output.channels = ch;
  output.data.resize(static_cast<std::size_t>(rr) * cc * ch);

  for (int ii = 0; ii < rr; ++ii) {
    std::size_t send_offset = (static_cast<std::size_t>(ii) * local_cols * ch);
    std::size_t recv_offset = (static_cast<std::size_t>(ii) * cc * ch);

    MPI_Allgatherv(local_out.data() + send_offset, counts[rank], MPI_BYTE, output.data.data() + recv_offset,
                   counts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
  }

  return true;
}

bool EgorovaLGaussFilterVertMPI::PostProcessingImpl() {
  return true;
}

}  // namespace egorova_l_gauss_filter_vert
