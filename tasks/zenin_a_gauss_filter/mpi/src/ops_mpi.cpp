#include "zenin_a_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "zenin_a_gauss_filter/common/include/common.hpp"

namespace zenin_a_gauss_filter {

namespace {

constexpr int kKernelSum = 16;

struct BlockInfo {
  int my_h = 0, my_w = 0;
  int start_y = 0, start_x = 0;
};

int Clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(hi, v));
}

std::uint8_t Clampu8(int v) {
  return static_cast<std::uint8_t>(Clampi(v, 0, 255));
}

std::uint8_t GetLocal(const std::vector<std::uint8_t> &buf, int local_w_with_halo, int ch, int x, int y, int c) {
  const int idx = ((y * local_w_with_halo + x) * ch) + c;
  return buf[idx];
}

BlockInfo CalcBlock(int pr, int pc, int h, int w, int grid_r, int grid_c) {
  const int base_h = h / grid_r;
  const int base_w = w / grid_c;
  const int extra_h = h % grid_r;
  const int extra_w = w % grid_c;

  BlockInfo b;
  b.my_h = base_h + (pr < extra_h ? 1 : 0);
  b.my_w = base_w + (pc < extra_w ? 1 : 0);

  b.start_y = (pr * base_h) + std::min(pr, extra_h);
  b.start_x = (pc * base_w) + std::min(pc, extra_w);
  return b;
}

}  // namespace

ZeninAGaussFilterMPI::ZeninAGaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninAGaussFilterMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool ok = true;
  if (rank == 0) {
    const auto &in = GetInput();
    const std::size_t need = static_cast<std::size_t>(in.width) * in.height * in.channels;

    ok = (in.width > 0) && (in.height > 0) && (in.channels == 1 || in.channels == 3) && (in.pixels.size() == need);
  }

  int ok_int = ok ? 1 : 0;

  MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok_int == 1;
}

bool ZeninAGaussFilterMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  if (rank == 0) {
    const auto &in = GetInput();
    width_ = in.width;
    height_ = in.height;
    channels_ = in.channels;
  }

  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::array<int, 2> dims{0, 0};
  MPI_Dims_create(proc_num_, 2, dims.data());
  grid_rows_ = dims[0];
  grid_cols_ = dims[1];

  block_h_ = height_ / grid_rows_;
  block_w_ = width_ / grid_cols_;
  extra_h_ = height_ % grid_rows_;
  extra_w_ = width_ % grid_cols_;

  if (rank == 0) {
    auto &out = GetOutput();
    out.height = height_;
    out.width = width_;
    out.channels = channels_;
    out.pixels.assign(static_cast<std::size_t>(width_) * height_ * channels_, 0);
  }

  return true;
}

bool ZeninAGaussFilterMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int pr = rank / grid_cols_;
  const int pc = rank % grid_cols_;

  const BlockInfo b = CalcBlock(pr, pc, height_, width_, grid_rows_, grid_cols_);
  const int my_h = b.my_h;
  const int my_w = b.my_w;

  const int halo = 1;
  const int lw = my_w + (2 * halo);
  const int lh = my_h + (2 * halo);

  std::vector<std::uint8_t> local_in(static_cast<std::size_t>(lh) * lw * channels_, 0);
  std::vector<std::uint8_t> local_out(static_cast<std::size_t>(my_h) * my_w * channels_, 0);
  auto global_idx = [&](int gx, int gy, int c) -> std::size_t {
    return ((static_cast<std::size_t>(gy) * width_ + gx) * channels_) + c;
  };

  if (rank == 0) {
    const auto &img = GetInput();

    auto fill_expanded_for = [&](const BlockInfo &bb, std::vector<std::uint8_t> &dst) {
      const int hh = bb.my_h;
      const int ww = bb.my_w;
      const int dst_w = ww + (2 * halo);
      const int dst_h = hh + (2 * halo);

      dst.assign(static_cast<std::size_t>(dst_h) * dst_w * channels_, 0);

      for (int ly = -halo; ly < hh + halo; ++ly) {
        for (int lx = -halo; lx < ww + halo; ++lx) {
          int gy = bb.start_y + ly;
          int gx = bb.start_x + lx;

          gy = std::max(0, std::min(height_ - 1, gy));
          gx = std::max(0, std::min(width_ - 1, gx));

          const int dy = ly + halo;
          const int dx = lx + halo;

          for (int chan = 0; chan < channels_; ++chan) {
            dst[((dy * dst_w + dx) * channels_) + chan] = img.pixels[global_idx(gx, gy, chan)];
          }
        }
      }
    };

    {
      std::vector<std::uint8_t> tmp;
      fill_expanded_for(b, tmp);
      local_in = std::move(tmp);
    }

    for (int ron = 1; ron < proc_num_; ++ron) {
      const int rpr = ron / grid_cols_;
      const int rpc = ron % grid_cols_;
      const BlockInfo rb = CalcBlock(rpr, rpc, height_, width_, grid_rows_, grid_cols_);

      std::vector<std::uint8_t> pack;
      fill_expanded_for(rb, pack);
      MPI_Send(pack.data(), static_cast<int>(pack.size()), MPI_UNSIGNED_CHAR, ron, 200, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_in.data(), static_cast<int>(local_in.size()), MPI_UNSIGNED_CHAR, 0, 200, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  for (int yd = 0; yd < my_h; ++yd) {
    const int ly = yd + halo;
    for (int xd = 0; xd < my_w; ++xd) {
      const int lx = xd + halo;
      for (int chan = 0; chan < channels_; ++chan) {
        const int v00 = static_cast<int>(GetLocal(local_in, lw, channels_, lx - 1, ly - 1, chan));
        const int v01 = static_cast<int>(GetLocal(local_in, lw, channels_, lx, ly - 1, chan));
        const int v02 = static_cast<int>(GetLocal(local_in, lw, channels_, lx + 1, ly - 1, chan));

        const int v10 = static_cast<int>(GetLocal(local_in, lw, channels_, lx - 1, ly, chan));
        const int v11 = static_cast<int>(GetLocal(local_in, lw, channels_, lx, ly, chan));
        const int v12 = static_cast<int>(GetLocal(local_in, lw, channels_, lx + 1, ly, chan));

        const int v20 = static_cast<int>(GetLocal(local_in, lw, channels_, lx - 1, ly + 1, chan));
        const int v21 = static_cast<int>(GetLocal(local_in, lw, channels_, lx, ly + 1, chan));
        const int v22 = static_cast<int>(GetLocal(local_in, lw, channels_, lx + 1, ly + 1, chan));

        int sum = 0;
        sum += v00 * 1;
        sum += v01 * 2;
        sum += v02 * 1;
        sum += v10 * 2;
        sum += v11 * 4;
        sum += v12 * 2;
        sum += v20 * 1;
        sum += v21 * 2;
        sum += v22 * 1;

        const int res = (sum + (kKernelSum / 2)) / kKernelSum;
        local_out[((yd * my_w + xd) * channels_) + chan] = Clampu8(res);
      }
    }
  }

  std::vector<std::uint8_t> final_image(static_cast<std::size_t>(width_) * height_ * channels_, 0);

  if (rank == 0) {
    for (int yd = 0; yd < my_h; ++yd) {
      for (int xd = 0; xd < my_w; ++xd) {
        const int gy = b.start_y + yd;
        const int gx = b.start_x + xd;
        for (int chan = 0; chan < channels_; ++chan) {
          final_image[((gy * width_ + gx) * channels_) + chan] = local_out[((yd * my_w + xd) * channels_) + chan];
        }
      }
    }

    for (int src = 1; src < proc_num_; ++src) {
      const int spr = src / grid_cols_;
      const int spc = src % grid_cols_;
      const BlockInfo sb = CalcBlock(spr, spc, height_, width_, grid_rows_, grid_cols_);

      std::vector<std::uint8_t> recv(static_cast<std::size_t>(sb.my_h) * sb.my_w * channels_);
      MPI_Recv(recv.data(), static_cast<int>(recv.size()), MPI_UNSIGNED_CHAR, src, 500, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      for (int yd = 0; yd < sb.my_h; ++yd) {
        for (int xd = 0; xd < sb.my_w; ++xd) {
          const int gy = sb.start_y + yd;
          const int gx = sb.start_x + xd;
          for (int chan = 0; chan < channels_; ++chan) {
            final_image[((gy * width_ + gx) * channels_) + chan] = recv[((yd * sb.my_w + xd) * channels_) + chan];
          }
        }
      }
    }
  } else {
    MPI_Send(local_out.data(), static_cast<int>(local_out.size()), MPI_UNSIGNED_CHAR, 0, 500, MPI_COMM_WORLD);
  }

  MPI_Bcast(final_image.data(), static_cast<int>(final_image.size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  GetOutput() = OutType{height_, width_, channels_, std::move(final_image)};
  return true;
}

bool ZeninAGaussFilterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_gauss_filter
