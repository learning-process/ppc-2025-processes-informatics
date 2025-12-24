#include "sannikov_i_shtrassen_algorithm/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"

namespace sannikov_i_shtrassen_algorithm {

namespace {

using Matrix = std::vector<std::vector<double>>;
using Flat = std::vector<double>;

constexpr std::size_t kClassicThreshold = 1;

std::size_t NextPow2(std::size_t value) {
  std::size_t pow2 = 1;
  while (pow2 < value) {
    pow2 <<= 1U;
  }
  return pow2;
}

std::size_t Idx(std::size_t row, std::size_t col, std::size_t stride) {
  return (row * stride) + col;
}

void PackPadToFlat(const Matrix &src, std::size_t n0, std::size_t m, Flat *out_flat) {
  out_flat->assign(m * m, 0.0);
  for (std::size_t row = 0; row < n0; ++row) {
    for (std::size_t col = 0; col < n0; ++col) {
      (*out_flat)[Idx(row, col, m)] = src[row][col];
    }
  }
}

Matrix UnpackCropFromFlat(const Flat &flat, std::size_t m, std::size_t n0) {
  Matrix out(n0, std::vector<double>(n0, 0.0));
  for (std::size_t row = 0; row < n0; ++row) {
    for (std::size_t col = 0; col < n0; ++col) {
      out[row][col] = flat[Idx(row, col, m)];
    }
  }
  return out;
}

void AddFlat(const Flat &a, const Flat &b, Flat *out) {
  const auto nn = a.size();
  out->assign(nn, 0.0);
  for (std::size_t idx = 0; idx < nn; ++idx) {
    (*out)[idx] = a[idx] + b[idx];
  }
}

void SubFlat(const Flat &a, const Flat &b, Flat *out) {
  const auto nn = a.size();
  out->assign(nn, 0.0);
  for (std::size_t idx = 0; idx < nn; ++idx) {
    (*out)[idx] = a[idx] - b[idx];
  }
}

Flat MultiplyClassicFlat(const Flat &a, const Flat &b, std::size_t n) {
  Flat c(n * n, 0.0);
  for (std::size_t row = 0; row < n; ++row) {
    double *crow = &c[Idx(row, 0, n)];
    for (std::size_t mid = 0; mid < n; ++mid) {
      const double aik = a[Idx(row, mid, n)];
      const double *brow = &b[Idx(mid, 0, n)];
      for (std::size_t col = 0; col < n; ++col) {
        crow[col] += aik * brow[col];
      }
    }
  }
  return c;
}

Flat ExtractBlock(const Flat &src, std::size_t src_n, std::size_t row0, std::size_t col0, std::size_t blk_n) {
  Flat blk(blk_n * blk_n, 0.0);
  for (std::size_t row = 0; row < blk_n; ++row) {
    for (std::size_t col = 0; col < blk_n; ++col) {
      blk[Idx(row, col, blk_n)] = src[Idx(row + row0, col + col0, src_n)];
    }
  }
  return blk;
}

void PlaceBlock(const Flat &blk, std::size_t blk_n, Flat *dst, std::size_t dst_n, std::size_t row0, std::size_t col0) {
  for (std::size_t row = 0; row < blk_n; ++row) {
    for (std::size_t col = 0; col < blk_n; ++col) {
      (*dst)[Idx(row + row0, col + col0, dst_n)] = blk[Idx(row, col, blk_n)];
    }
  }
}

struct Frame {
  Flat a;
  Flat b;
  std::size_t n = 0;

  bool has_parent = false;
  std::size_t parent_idx = 0;
  int parent_slot = 0;

  int stage = 0;
  Flat res;

  Flat a11, a12, a21, a22;
  Flat b11, b12, b21, b22;

  Flat m1, m2, m3, m4, m5, m6, m7;
};

void AssignToParent(std::vector<Frame> *stack, const Frame &child) {
  if (!child.has_parent) {
    return;
  }

  Frame &parent = (*stack)[child.parent_idx];
  const int slot = child.parent_slot;

  if (slot == 1) {
    parent.m1 = child.res;
  }
  if (slot == 2) {
    parent.m2 = child.res;
  }
  if (slot == 3) {
    parent.m3 = child.res;
  }
  if (slot == 4) {
    parent.m4 = child.res;
  }
  if (slot == 5) {
    parent.m5 = child.res;
  }
  if (slot == 6) {
    parent.m6 = child.res;
  }
  if (slot == 7) {
    parent.m7 = child.res;
  }
}

bool IsLeaf(const Frame &frame) {
  return frame.n <= kClassicThreshold;
}

void SplitIfNeeded(Frame *frame) {
  if (frame->stage != 0) {
    return;
  }

  const auto half = frame->n / 2;

  frame->a11 = ExtractBlock(frame->a, frame->n, 0, 0, half);
  frame->a12 = ExtractBlock(frame->a, frame->n, 0, half, half);
  frame->a21 = ExtractBlock(frame->a, frame->n, half, 0, half);
  frame->a22 = ExtractBlock(frame->a, frame->n, half, half, half);
  frame->b11 = ExtractBlock(frame->b, frame->n, 0, 0, half);
  frame->b12 = ExtractBlock(frame->b, frame->n, 0, half, half);
  frame->b21 = ExtractBlock(frame->b, frame->n, half, 0, half);
  frame->b22 = ExtractBlock(frame->b, frame->n, half, half, half);

  frame->stage = 1;
}

void BuildChildOperands(const Frame &parent, int slot, Flat *left, Flat *right) {
  if (slot == 1) {
    AddFlat(parent.a11, parent.a22, left);
    AddFlat(parent.b11, parent.b22, right);
    return;
  }
  if (slot == 2) {
    AddFlat(parent.a21, parent.a22, left);
    *right = parent.b11;
    return;
  }
  if (slot == 3) {
    *left = parent.a11;
    SubFlat(parent.b12, parent.b22, right);
    return;
  }
  if (slot == 4) {
    *left = parent.a22;
    SubFlat(parent.b21, parent.b11, right);
    return;
  }
  if (slot == 5) {
    AddFlat(parent.a11, parent.a12, left);
    *right = parent.b22;
    return;
  }
  if (slot == 6) {
    SubFlat(parent.a21, parent.a11, left);
    AddFlat(parent.b11, parent.b12, right);
    return;
  }
  SubFlat(parent.a12, parent.a22, left);
  AddFlat(parent.b21, parent.b22, right);
}

Frame MakeChild(const Frame &parent, std::size_t parent_idx, int slot) {
  Flat left;
  Flat right;
  BuildChildOperands(parent, slot, &left, &right);

  Frame child;
  child.a = std::move(left);
  child.b = std::move(right);
  child.n = parent.n / 2;

  child.has_parent = true;
  child.parent_idx = parent_idx;
  child.parent_slot = slot;

  child.stage = 0;
  return child;
}

void CombineOnFrame(Frame *frame) {
  const auto half = frame->n / 2;
  const auto kk = half * half;

  Flat c11(kk, 0.0);
  Flat c12(kk, 0.0);
  Flat c21(kk, 0.0);
  Flat c22(kk, 0.0);

  for (std::size_t idx = 0; idx < kk; ++idx) {
    c11[idx] = frame->m1[idx] + frame->m4[idx] - frame->m5[idx] + frame->m7[idx];
  }
  for (std::size_t idx = 0; idx < kk; ++idx) {
    c12[idx] = frame->m3[idx] + frame->m5[idx];
  }
  for (std::size_t idx = 0; idx < kk; ++idx) {
    c21[idx] = frame->m2[idx] + frame->m4[idx];
  }
  for (std::size_t idx = 0; idx < kk; ++idx) {
    c22[idx] = frame->m1[idx] - frame->m2[idx] + frame->m3[idx] + frame->m6[idx];
  }

  frame->res.assign(frame->n * frame->n, 0.0);
  PlaceBlock(c11, half, &frame->res, frame->n, 0, 0);
  PlaceBlock(c12, half, &frame->res, frame->n, 0, half);
  PlaceBlock(c21, half, &frame->res, frame->n, half, 0);
  PlaceBlock(c22, half, &frame->res, frame->n, half, half);
}

bool NeedSpawnChild(const Frame &frame) {
  return (frame.stage >= 1) && (frame.stage <= 7);
}

bool ReadyToCombine(const Frame &frame) {
  return frame.stage == 8;
}

Flat ShtrassenIterative(const Flat &a0, const Flat &b0, std::size_t n0) {
  std::vector<Frame> frames;
  frames.reserve(64);

  Frame root;
  root.a = a0;
  root.b = b0;
  root.n = n0;
  root.has_parent = false;
  root.stage = 0;
  frames.push_back(root);

  while (!frames.empty()) {
    Frame &cur = frames.back();

    if (IsLeaf(cur)) {
      cur.res = MultiplyClassicFlat(cur.a, cur.b, cur.n);

      const Frame finished = cur;
      frames.pop_back();

      if (frames.empty()) {
        return finished.res;
      }

      AssignToParent(&frames, finished);
      continue;
    }

    SplitIfNeeded(&cur);

    if (NeedSpawnChild(cur)) {
      const int slot = cur.stage;
      const std::size_t parent_idx = frames.size() - 1;

      Frame child = MakeChild(cur, parent_idx, slot);
      cur.stage += 1;
      frames.push_back(std::move(child));
      continue;
    }

    if (!ReadyToCombine(cur)) {
      cur.stage = 8;
      continue;
    }

    CombineOnFrame(&cur);

    const Frame finished = cur;
    frames.pop_back();

    if (frames.empty()) {
      return finished.res;
    }

    AssignToParent(&frames, finished);
  }

  return Flat{};
}

}  // namespace

SannikovIShtrassenAlgorithmSEQ::SannikovIShtrassenAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

bool SannikovIShtrassenAlgorithmSEQ::ValidationImpl() {
  (void)this;
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

  const std::size_t nn = mat_a.size();
  for (const auto &row : mat_a) {
    if (row.size() != nn) {
      return false;
    }
  }
  for (const auto &row : mat_b) {
    if (row.size() != nn) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmSEQ::PreProcessingImpl() {
  (void)this;
  GetOutput().clear();
  return true;
}

bool SannikovIShtrassenAlgorithmSEQ::RunImpl() {
  (void)this;
  const auto &input = GetInput();
  const auto &a_in = std::get<0>(input);
  const auto &b_in = std::get<1>(input);

  const std::size_t n0 = a_in.size();
  const std::size_t m = NextPow2(n0);

  Flat a_flat;
  Flat b_flat;
  PackPadToFlat(a_in, n0, m, &a_flat);
  PackPadToFlat(b_in, n0, m, &b_flat);

  const Flat c_flat = ShtrassenIterative(a_flat, b_flat, m);
  GetOutput() = UnpackCropFromFlat(c_flat, m, n0);

  return !GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmSEQ::PostProcessingImpl() {
  (void)this;
  return !GetOutput().empty();
}

}  // namespace sannikov_i_shtrassen_algorithm
