#include "akimov_i_star/seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "akimov_i_star/common/include/common.hpp"

namespace akimov_i_star {

namespace {

bool CheckPrefix(const char *data, std::size_t n, std::size_t i, const char *prefix, std::size_t prefix_len) {
  if (i + prefix_len > n) {
    return false;
  }
  for (std::size_t k = 0; k < prefix_len; ++k) {
    if (data[i + k] != prefix[k]) {
      return false;
    }
  }
  return true;
}

std::size_t SkipWhitespace(const char *data, std::size_t n, std::size_t pos) {
  while (pos < n && (data[pos] == ' ' || data[pos] == '\t' || data[pos] == '\r')) {
    ++pos;
  }
  return pos;
}

bool ProcessLineForZeroDst(const char *data, std::size_t n, std::size_t &i) {
  const char prefix[] = "send:";
  const std::size_t prefix_len = 5;

  if (!CheckPrefix(data, n, i, prefix, prefix_len)) {
    return false;
  }

  std::size_t j = i + prefix_len;

  while (j < n && data[j] != '\n' && data[j] != ':') {
    ++j;
  }

  if (j >= n || data[j] != ':') {
    return false;
  }

  std::size_t dst_start = j + 1;
  dst_start = SkipWhitespace(data, n, dst_start);

  if (dst_start >= n) {
    return false;
  }

  if (data[dst_start] != '0') {
    return false;
  }

  std::size_t after = dst_start + 1;
  after = SkipWhitespace(data, n, after);

  if (after < n && data[after] == ':') {
    return true;
  }

  return false;
}

int CountDstZeroFromBuffer(const InType &buf) {
  const char *data = buf.empty() ? nullptr : buf.data();
  std::size_t n = buf.size();
  if (n == 0 || data == nullptr) {
    return 0;
  }

  int count = 0;
  std::size_t i = 0;

  while (i < n) {
    if (data[i] == 's') {
      std::size_t current_pos = i;
      if (ProcessLineForZeroDst(data, n, current_pos)) {
        ++count;
      }

      while (i < n && data[i] != '\n') {
        ++i;
      }
      if (i < n && data[i] == '\n') {
        ++i;
      }
    } else if (data[i] == '\n') {
      ++i;
    } else {
      while (i < n && data[i] != '\n') {
        ++i;
      }
      if (i < n && data[i] == '\n') {
        ++i;
      }
    }
  }

  return count;
}

}  // namespace

AkimovIStarSEQ::AkimovIStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkimovIStarSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool AkimovIStarSEQ::PreProcessingImpl() {
  input_buffer_ = GetInput();
  received_count_ = CountDstZeroFromBuffer(input_buffer_);
  return true;
}

bool AkimovIStarSEQ::RunImpl() {
  GetOutput() = received_count_;
  return true;
}

bool AkimovIStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace akimov_i_star
