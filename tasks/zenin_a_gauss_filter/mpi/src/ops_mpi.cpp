#include "zenin_a_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "zenin_a_gauss_filter/common/include/common.hpp"

namespace zenin_a_gauss_filter {
inline int clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(hi, v));
}
inline std::uint8_t clampu8(int v) {
  return static_cast<std::uint8_t>(clampi(v, 0, 255));
}

constexpr int K[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
constexpr int K_SUM = 16;

inline std::uint8_t getLocal(const std::vector<std::uint8_t> &buf, int local_w_with_halo, int ch, int x, int y, int c) {
  const int idx = ((y * local_w_with_halo + x) * ch) + c;
  return buf[idx];
}

inline void setLocal(std::vector<std::uint8_t> &buf, int local_w_with_halo, int ch, int x, int y, int c,
                     std::uint8_t v) {
  const int idx = ((y * local_w_with_halo + x) * ch) + c;
  buf[idx] = v;
}

struct BlockInfo {
  int my_h = 0, my_w = 0;
  int start_y = 0, start_x = 0;
};

inline BlockInfo calcBlock(int pr, int pc, int H, int W, int grid_r, int grid_c) {
  const int base_h = H / grid_r;
  const int base_w = W / grid_c;
  const int extra_h = H % grid_r;
  const int extra_w = W % grid_c;

  BlockInfo b;
  b.my_h = base_h + (pr < extra_h ? 1 : 0);
  b.my_w = base_w + (pc < extra_w ? 1 : 0);

  b.start_y = pr * base_h + std::min(pr, extra_h);
  b.start_x = pc * base_w + std::min(pc, extra_w);
  return b;
}

ZeninAGaussFilterMPI::ZeninAGaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninAGaussFilterMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ok = 1;
  if (rank == 0) {
    const auto &in = GetInput();
    const std::size_t need = static_cast<std::size_t>(in.width) * in.height * in.channels;

    ok = (in.width > 0) && (in.height > 0) && (in.channels == 1 || in.channels == 3) && (in.pixels.size() == need);
  }

  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok == 1;
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

  int dims[2] = {0, 0};
  MPI_Dims_create(proc_num_, 2, dims);
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

  const BlockInfo b = calcBlock(pr, pc, height_, width_, grid_rows_, grid_cols_);
  const int my_h = b.my_h;
  const int my_w = b.my_w;

  const int halo = 1;
  const int lw = my_w + 2 * halo;  // local width with halo
  const int lh = my_h + 2 * halo;  // local height with halo

  std::vector<std::uint8_t> local_in(static_cast<std::size_t>(lh) * lw * channels_, 0);
  std::vector<std::uint8_t> local_out(static_cast<std::size_t>(my_h) * my_w * channels_, 0);

  if (rank == 0) {
    const auto &img = GetInput();

    for (int y = 0; y < my_h; ++y) {
      for (int x = 0; x < my_w; ++x) {
        const int gy = b.start_y + y;
        const int gx = b.start_x + x;
        for (int c = 0; c < channels_; ++c) {
          const int gidx = ((gy * width_ + gx) * channels_) + c;
          setLocal(local_in, lw, channels_, x + halo, y + halo, c, img.pixels[gidx]);
        }
      }
    }

    for (int r = 1; r < proc_num_; ++r) {
      const int rpr = r / grid_cols_;
      const int rpc = r % grid_cols_;
      const BlockInfo rb = calcBlock(rpr, rpc, height_, width_, grid_rows_, grid_cols_);

      std::vector<std::uint8_t> pack(static_cast<std::size_t>(rb.my_h) * rb.my_w * channels_);
      for (int y = 0; y < rb.my_h; ++y) {
        for (int x = 0; x < rb.my_w; ++x) {
          const int gy = rb.start_y + y;
          const int gx = rb.start_x + x;
          for (int c = 0; c < channels_; ++c) {
            const int gidx = ((gy * width_ + gx) * channels_) + c;
            pack[((y * rb.my_w + x) * channels_) + c] = img.pixels[gidx];
          }
        }
      }
      MPI_Send(pack.data(), static_cast<int>(pack.size()), MPI_UNSIGNED_CHAR, r, 200, MPI_COMM_WORLD);
    }
  } else {
    std::vector<std::uint8_t> pack(static_cast<std::size_t>(my_h) * my_w * channels_);
    MPI_Recv(pack.data(), static_cast<int>(pack.size()), MPI_UNSIGNED_CHAR, 0, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int y = 0; y < my_h; ++y) {
      for (int x = 0; x < my_w; ++x) {
        for (int c = 0; c < channels_; ++c) {
          setLocal(local_in, lw, channels_, x + halo, y + halo, c, pack[((y * my_w + x) * channels_) + c]);
        }
      }
    }
  }

  const int up = (pr > 0) ? (rank - grid_cols_) : MPI_PROC_NULL;
  const int down = (pr < grid_rows_ - 1) ? (rank + grid_cols_) : MPI_PROC_NULL;
  const int left = (pc > 0) ? (rank - 1) : MPI_PROC_NULL;
  const int right = (pc < grid_cols_ - 1) ? (rank + 1) : MPI_PROC_NULL;

  const int row_bytes = lw * channels_;

  if (up != MPI_PROC_NULL) {
    MPI_Sendrecv(local_in.data() + (1 * row_bytes), row_bytes, MPI_UNSIGNED_CHAR, up, 300,
                 local_in.data() + (0 * row_bytes), row_bytes, MPI_UNSIGNED_CHAR, up, 301, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
  } else {
    std::copy(local_in.begin() + (1 * row_bytes), local_in.begin() + (2 * row_bytes),
              local_in.begin() + (0 * row_bytes));
  }

  if (down != MPI_PROC_NULL) {
    MPI_Sendrecv(local_in.data() + ((lh - 2) * row_bytes), row_bytes, MPI_UNSIGNED_CHAR, down, 301,
                 local_in.data() + ((lh - 1) * row_bytes), row_bytes, MPI_UNSIGNED_CHAR, down, 300, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
  } else {
    std::copy(local_in.begin() + ((lh - 2) * row_bytes), local_in.begin() + ((lh - 1) * row_bytes),
              local_in.begin() + ((lh - 1) * row_bytes));
  }

  std::vector<std::uint8_t> send_col(static_cast<std::size_t>(lh) * channels_);
  std::vector<std::uint8_t> recv_col(static_cast<std::size_t>(lh) * channels_);

  if (left != MPI_PROC_NULL) {
    for (int y = 0; y < lh; ++y) {
      for (int c = 0; c < channels_; ++c) {
        send_col[(y * channels_) + c] = getLocal(local_in, lw, channels_, 1, y, c);
      }
    }

    MPI_Sendrecv(send_col.data(), static_cast<int>(send_col.size()), MPI_UNSIGNED_CHAR, left, 400, recv_col.data(),
                 static_cast<int>(recv_col.size()), MPI_UNSIGNED_CHAR, left, 401, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int y = 0; y < lh; ++y) {
      for (int c = 0; c < channels_; ++c) {
        setLocal(local_in, lw, channels_, 0, y, c, recv_col[(y * channels_) + c]);
      }
    }
  } else {
    for (int y = 0; y < lh; ++y) {
      for (int c = 0; c < channels_; ++c) {
        setLocal(local_in, lw, channels_, 0, y, c, getLocal(local_in, lw, channels_, 1, y, c));
      }
    }
  }

  if (right != MPI_PROC_NULL) {
    for (int y = 0; y < lh; ++y) {
      for (int c = 0; c < channels_; ++c) {
        send_col[(y * channels_) + c] = getLocal(local_in, lw, channels_, lw - 2, y, c);
      }
    }

    MPI_Sendrecv(send_col.data(), static_cast<int>(send_col.size()), MPI_UNSIGNED_CHAR, right, 401, recv_col.data(),
                 static_cast<int>(recv_col.size()), MPI_UNSIGNED_CHAR, right, 400, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int y = 0; y < lh; ++y) {
      for (int c = 0; c < channels_; ++c) {
        setLocal(local_in, lw, channels_, lw - 1, y, c, recv_col[(y * channels_) + c]);
      }
    }
  } else {
    for (int y = 0; y < lh; ++y) {
      for (int c = 0; c < channels_; ++c) {
        setLocal(local_in, lw, channels_, lw - 1, y, c, getLocal(local_in, lw, channels_, lw - 2, y, c));
      }
    }
  }

  for (int y = 0; y < my_h; ++y) {
    const int ly = y + halo;
    for (int x = 0; x < my_w; ++x) {
      const int lx = x + halo;
      for (int c = 0; c < channels_; ++c) {
        int sum = 0;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            sum += K[dy + 1][dx + 1] * static_cast<int>(getLocal(local_in, lw, channels_, lx + dx, ly + dy, c));
          }
        }
        const int res = (sum + (K_SUM / 2)) / K_SUM;  // (sum + 8) / 16
        local_out[((y * my_w + x) * channels_) + c] = clampu8(res);
      }
    }
  }

  std::vector<std::uint8_t> final_image(static_cast<std::size_t>(width_) * height_ * channels_, 0);

  if (rank == 0) {
    for (int y = 0; y < my_h; ++y) {
      for (int x = 0; x < my_w; ++x) {
        const int gy = b.start_y + y;
        const int gx = b.start_x + x;
        for (int c = 0; c < channels_; ++c) {
          final_image[((gy * width_ + gx) * channels_) + c] = local_out[((y * my_w + x) * channels_) + c];
        }
      }
    }

    for (int src = 1; src < proc_num_; ++src) {
      const int spr = src / grid_cols_;
      const int spc = src % grid_cols_;
      const BlockInfo sb = calcBlock(spr, spc, height_, width_, grid_rows_, grid_cols_);

      std::vector<std::uint8_t> recv(static_cast<std::size_t>(sb.my_h) * sb.my_w * channels_);
      MPI_Recv(recv.data(), static_cast<int>(recv.size()), MPI_UNSIGNED_CHAR, src, 500, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      for (int y = 0; y < sb.my_h; ++y) {
        for (int x = 0; x < sb.my_w; ++x) {
          const int gy = sb.start_y + y;
          const int gx = sb.start_x + x;
          for (int c = 0; c < channels_; ++c) {
            final_image[((gy * width_ + gx) * channels_) + c] = recv[((y * sb.my_w + x) * channels_) + c];
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
