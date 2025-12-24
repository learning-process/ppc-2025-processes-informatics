#include "sannikov_i_shtrassen_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"

namespace sannikov_i_shtrassen_algorithm {

SannikovIShtrassenAlgorithmMPI::SannikovIShtrassenAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

bool SannikovIShtrassenAlgorithmMPI::ValidationImpl() {
  const auto &input = GetInput();
  const auto &mat_a = std::get<0>(input);
  const auto &mat_b = std::get<1>(input);

  if (mat_a.empty() || mat_b.empty()) {
    return false;
  }
  if (mat_a.size() != mat_b.size()) {
    return false;
  }
  if (mat_a.front().empty() || mat_b.front().empty()) {
    return false;
  }

  const auto n = mat_a.size();
  for (const auto &row : mat_a) {
    if (row.size() != n) {
      return false;
    }
  }
  for (const auto &row : mat_b) {
    if (row.size() != n) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

using Flat = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

std::size_t NextPow2(std::size_t value) {
  std::size_t pow2 = 1;
  while (pow2 < value) {
    pow2 <<= 1U;
  }
  return pow2;
}

bool SizeOkU64(std::uint64_t n64) {
  if (n64 == 0U) {
    return false;
  }
  if (n64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  return true;
}

std::size_t Idx(std::size_t row, std::size_t col, std::size_t ld) {
  return (row * ld) + col;
}

void PadToFlatOnRoot(const Matrix &src, int n0, int m, Flat *flat) {
  flat->assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  const auto mm = static_cast<std::size_t>(m);
  for (int row = 0; row < n0; ++row) {
    for (int col = 0; col < n0; ++col) {
      const auto rr = static_cast<std::size_t>(row);
      const auto cc = static_cast<std::size_t>(col);
      (*flat)[Idx(rr, cc, mm)] = src[rr][cc];
    }
  }
}

void ExtractBlock(const Flat &src, int m, int block_row, int block_col, int k, Flat *block) {
  block->assign(static_cast<std::size_t>(k) * static_cast<std::size_t>(k), 0.0);

  const auto mm = static_cast<std::size_t>(m);
  const auto kk = static_cast<std::size_t>(k);
  const auto ro = static_cast<std::size_t>(block_row) * static_cast<std::size_t>(k);
  const auto co = static_cast<std::size_t>(block_col) * static_cast<std::size_t>(k);

  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < k; ++col) {
      const auto rr = ro + static_cast<std::size_t>(row);
      const auto cc = co + static_cast<std::size_t>(col);
      (*block)[Idx(static_cast<std::size_t>(row), static_cast<std::size_t>(col), kk)] = src[Idx(rr, cc, mm)];
    }
  }
}

void AddVec(const Flat &a, const Flat &b, Flat *out) {
  const auto n = a.size();
  out->assign(n, 0.0);
  for (std::size_t ii = 0; ii < n; ++ii) {
    (*out)[ii] = a[ii] + b[ii];
  }
}

void SubVec(const Flat &a, const Flat &b, Flat *out) {
  const auto n = a.size();
  out->assign(n, 0.0);
  for (std::size_t ii = 0; ii < n; ++ii) {
    (*out)[ii] = a[ii] - b[ii];
  }
}

void MulClassic(const Flat &a, const Flat &b, int k, Flat *out) {
  out->assign(static_cast<std::size_t>(k) * static_cast<std::size_t>(k), 0.0);
  const auto kk = static_cast<std::size_t>(k);

  for (int row = 0; row < k; ++row) {
    const auto rr = static_cast<std::size_t>(row);
    for (int mid = 0; mid < k; ++mid) {
      const auto mid_sz = static_cast<std::size_t>(mid);
      const double aik = a[Idx(rr, mid_sz, kk)];
      for (int col = 0; col < k; ++col) {
        const auto cc = static_cast<std::size_t>(col);
        (*out)[Idx(rr, cc, kk)] += aik * b[Idx(mid_sz, cc, kk)];
      }
    }
  }
}

int OwnerForTask(int task_id, int comm_size) {
  if (comm_size <= 0) {
    return 0;
  }
  return (task_id - 1) % comm_size;
}

void ComputeOneStrassenTask(int task_id, const Flat &a11, const Flat &a12, const Flat &a21, const Flat &a22,
                            const Flat &b11, const Flat &b12, const Flat &b21, const Flat &b22, int k, Flat *out_m) {
  Flat tmp1;
  Flat tmp2;

  if (task_id == 1) {
    AddVec(a11, a22, &tmp1);
    AddVec(b11, b22, &tmp2);
    MulClassic(tmp1, tmp2, k, out_m);
    return;
  }
  if (task_id == 2) {
    AddVec(a21, a22, &tmp1);
    MulClassic(tmp1, b11, k, out_m);
    return;
  }
  if (task_id == 3) {
    SubVec(b12, b22, &tmp2);
    MulClassic(a11, tmp2, k, out_m);
    return;
  }
  if (task_id == 4) {
    SubVec(b21, b11, &tmp2);
    MulClassic(a22, tmp2, k, out_m);
    return;
  }
  if (task_id == 5) {
    AddVec(a11, a12, &tmp1);
    MulClassic(tmp1, b22, k, out_m);
    return;
  }
  if (task_id == 6) {
    SubVec(a21, a11, &tmp1);
    AddVec(b11, b12, &tmp2);
    MulClassic(tmp1, tmp2, k, out_m);
    return;
  }

  SubVec(a12, a22, &tmp1);
  AddVec(b21, b22, &tmp2);
  MulClassic(tmp1, tmp2, k, out_m);
}

void AssembleCOnRoot(const Flat &m1, const Flat &m2, const Flat &m3, const Flat &m4, const Flat &m5, const Flat &m6,
                     const Flat &m7, int m, int k, Flat *c_full) {
  c_full->assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  const auto mm = static_cast<std::size_t>(m);
  const auto kk = static_cast<std::size_t>(k);

  for (int row = 0; row < k; ++row) {
    const auto rr = static_cast<std::size_t>(row);
    for (int col = 0; col < k; ++col) {
      const auto cc = static_cast<std::size_t>(col);
      const auto id = Idx(rr, cc, kk);
      const double c11 = m1[id] + m4[id] - m5[id] + m7[id];
      const double c12 = m3[id] + m5[id];
      const double c21 = m2[id] + m4[id];
      const double c22 = m1[id] - m2[id] + m3[id] + m6[id];
      (*c_full)[Idx(rr, cc, mm)] = c11;
      (*c_full)[Idx(rr, cc + kk, mm)] = c12;
      (*c_full)[Idx(rr + kk, cc, mm)] = c21;
      (*c_full)[Idx(rr + kk, cc + kk, mm)] = c22;
    }
  }
}

bool FastPath1x1(int rank, const Matrix &a, const Matrix &b, Matrix *out) {
  double value = 0.0;
  if (rank == 0) {
    value = a[0][0] * b[0][0];
  }
  MPI_Bcast(&value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  *out = Matrix(1, std::vector<double>(1, value));
  return true;
}

void BroadcastFullPads(int rank, int m, const Matrix &a_in, const Matrix &b_in, int n0, Flat *a_full, Flat *b_full) {
  const auto total = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);
  if (rank == 0) {
    PadToFlatOnRoot(a_in, n0, m, a_full);
    PadToFlatOnRoot(b_in, n0, m, b_full);
  } else {
    a_full->assign(total, 0.0);
    b_full->assign(total, 0.0);
  }

  MPI_Bcast(a_full->data(), static_cast<int>(a_full->size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_full->data(), static_cast<int>(b_full->size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void SplitBlocks(const Flat &a_full, const Flat &b_full, int m, int k, Flat *a11, Flat *a12, Flat *a21, Flat *a22,
                 Flat *b11, Flat *b12, Flat *b21, Flat *b22) {
  ExtractBlock(a_full, m, 0, 0, k, a11);
  ExtractBlock(a_full, m, 0, 1, k, a12);
  ExtractBlock(a_full, m, 1, 0, k, a21);
  ExtractBlock(a_full, m, 1, 1, k, a22);

  ExtractBlock(b_full, m, 0, 0, k, b11);
  ExtractBlock(b_full, m, 0, 1, k, b12);
  ExtractBlock(b_full, m, 1, 0, k, b21);
  ExtractBlock(b_full, m, 1, 1, k, b22);
}

void ComputeLocalMi(int rank, int comm_size, int k, const Flat &a11, const Flat &a12, const Flat &a21, const Flat &a22,
                    const Flat &b11, const Flat &b12, const Flat &b21, const Flat &b22, Flat *m1_loc, Flat *m2_loc,
                    Flat *m3_loc, Flat *m4_loc, Flat *m5_loc, Flat *m6_loc, Flat *m7_loc) {
  for (int task_id = 1; task_id <= 7; ++task_id) {
    const int owner = OwnerForTask(task_id, comm_size);
    if (rank != owner) {
      continue;
    }

    Flat tmp_out;
    ComputeOneStrassenTask(task_id, a11, a12, a21, a22, b11, b12, b21, b22, k, &tmp_out);

    if (task_id == 1) {
      m1_loc->swap(tmp_out);
    } else if (task_id == 2) {
      m2_loc->swap(tmp_out);
    } else if (task_id == 3) {
      m3_loc->swap(tmp_out);
    } else if (task_id == 4) {
      m4_loc->swap(tmp_out);
    } else if (task_id == 5) {
      m5_loc->swap(tmp_out);
    } else if (task_id == 6) {
      m6_loc->swap(tmp_out);
    } else {
      m7_loc->swap(tmp_out);
    }
  }
}

void ReduceMiToRoot(std::size_t kk, const Flat &m1_loc, const Flat &m2_loc, const Flat &m3_loc, const Flat &m4_loc,
                    const Flat &m5_loc, const Flat &m6_loc, const Flat &m7_loc, Flat *m1, Flat *m2, Flat *m3, Flat *m4,
                    Flat *m5, Flat *m6, Flat *m7) {
  MPI_Reduce(m1_loc.data(), m1->data(), static_cast<int>(kk), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(m2_loc.data(), m2->data(), static_cast<int>(kk), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(m3_loc.data(), m3->data(), static_cast<int>(kk), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(m4_loc.data(), m4->data(), static_cast<int>(kk), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(m5_loc.data(), m5->data(), static_cast<int>(kk), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(m6_loc.data(), m6->data(), static_cast<int>(kk), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(m7_loc.data(), m7->data(), static_cast<int>(kk), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void RootCropToFlat(int rank, int m, int n0, const Flat &c_full, Flat *c_crop_flat) {
  const auto total = static_cast<std::size_t>(n0) * static_cast<std::size_t>(n0);
  if (rank != 0) {
    c_crop_flat->assign(total, 0.0);
    return;
  }

  c_crop_flat->assign(total, 0.0);

  const auto mm = static_cast<std::size_t>(m);
  const auto nn = static_cast<std::size_t>(n0);

  for (int row = 0; row < n0; ++row) {
    for (int col = 0; col < n0; ++col) {
      const auto rr = static_cast<std::size_t>(row);
      const auto cc = static_cast<std::size_t>(col);
      (*c_crop_flat)[Idx(rr, cc, nn)] = c_full[Idx(rr, cc, mm)];
    }
  }
}

Matrix FlatToMatrix(int n0, const Flat &flat) {
  Matrix out(static_cast<std::size_t>(n0), std::vector<double>(static_cast<std::size_t>(n0), 0.0));
  const auto nn = static_cast<std::size_t>(n0);
  for (int row = 0; row < n0; ++row) {
    for (int col = 0; col < n0; ++col) {
      const auto rr = static_cast<std::size_t>(row);
      const auto cc = static_cast<std::size_t>(col);
      out[rr][cc] = flat[Idx(rr, cc, nn)];
    }
  }
  return out;
}

}  // namespace

bool SannikovIShtrassenAlgorithmMPI::RunImpl() {
  const auto &input = GetInput();
  const auto &mat_a = std::get<0>(input);
  const auto &mat_b = std::get<1>(input);

  int rank = 0;
  int comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  std::uint64_t n0_64 = 0U;
  if (rank == 0) {
    n0_64 = static_cast<std::uint64_t>(mat_a.size());
  }
  MPI_Bcast(&n0_64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (!SizeOkU64(n0_64)) {
    return false;
  }

  const int n0 = static_cast<int>(n0_64);

  const auto m_sz = NextPow2(static_cast<std::size_t>(n0));
  if (m_sz > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  const int m = static_cast<int>(m_sz);

  if (m == 1) {
    Matrix out;
    (void)FastPath1x1(rank, mat_a, mat_b, &out);
    GetOutput() = out;
    return true;
  }

  Flat a_full;
  Flat b_full;
  BroadcastFullPads(rank, m, mat_a, mat_b, n0, &a_full, &b_full);

  const int k = m / 2;

  Flat a11;
  Flat a12;
  Flat a21;
  Flat a22;
  Flat b11;
  Flat b12;
  Flat b21;
  Flat b22;
  SplitBlocks(a_full, b_full, m, k, &a11, &a12, &a21, &a22, &b11, &b12, &b21, &b22);

  const auto kk = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);

  Flat m1_loc(kk, 0.0);
  Flat m2_loc(kk, 0.0);
  Flat m3_loc(kk, 0.0);
  Flat m4_loc(kk, 0.0);
  Flat m5_loc(kk, 0.0);
  Flat m6_loc(kk, 0.0);
  Flat m7_loc(kk, 0.0);

  ComputeLocalMi(rank, comm_size, k, a11, a12, a21, a22, b11, b12, b21, b22, &m1_loc, &m2_loc, &m3_loc, &m4_loc,
                 &m5_loc, &m6_loc, &m7_loc);

  Flat m1(kk, 0.0);
  Flat m2(kk, 0.0);
  Flat m3(kk, 0.0);
  Flat m4(kk, 0.0);
  Flat m5(kk, 0.0);
  Flat m6(kk, 0.0);
  Flat m7(kk, 0.0);

  ReduceMiToRoot(kk, m1_loc, m2_loc, m3_loc, m4_loc, m5_loc, m6_loc, m7_loc, &m1, &m2, &m3, &m4, &m5, &m6, &m7);

  Flat c_full;
  if (rank == 0) {
    AssembleCOnRoot(m1, m2, m3, m4, m5, m6, m7, m, k, &c_full);
  }

  Flat c_crop_flat;
  RootCropToFlat(rank, m, n0, c_full, &c_crop_flat);
  MPI_Bcast(c_crop_flat.data(), static_cast<int>(c_crop_flat.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = FlatToMatrix(n0, c_crop_flat);
  return !GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace sannikov_i_shtrassen_algorithm
