#include "akimov_i_star/seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "akimov_i_star/common/include/common.hpp"

namespace akimov_i_star {

namespace {

int CountDstZeroFromBuffer(const InType &buf) {
  const char *data = buf.empty() ? nullptr : buf.data();
  std::size_t n = buf.size();
  if (n == 0 || data == nullptr) {
    return 0;
  }

  const char prefix[] = "send:";
  const std::size_t prefix_len = 5;

  int count = 0;
  std::size_t i = 0;

  while (i + prefix_len <= n) {
    if (data[i] == 's') {
      bool is_prefix = true;
      for (std::size_t k = 0; k < prefix_len; ++k) {
        if (i + k >= n || data[i + k] != prefix[k]) {
          is_prefix = false;
          break;
        }
      }
      if (!is_prefix) {
        ++i;
        continue;
      }
      std::size_t j = i + prefix_len;
      while (j < n && data[j] != '\n' && data[j] != ':') {
        ++j;
      }
      if (j >= n || data[j] != ':') {
        while (i < n && data[i] != '\n') {
          ++i;
        }
        if (i < n && data[i] == '\n') {
          ++i;
        }
        continue;
      }
      std::size_t dst_start = j + 1;
      while (dst_start < n && (data[dst_start] == ' ' || data[dst_start] == '\t' || data[dst_start] == '\r')) {
        ++dst_start;
      }
      if (dst_start >= n) {
        break;
      }

      if (data[dst_start] == '0') {
        std::size_t after = dst_start + 1;
        while (after < n && (data[after] == ' ' || data[after] == '\t' || data[after] == '\r')) {
          ++after;
        }
        if (after < n && data[after] == ':') {
          ++count;
        }
      }

      while (i < n && data[i] != '\n') {
        ++i;
      }
      if (i < n && data[i] == '\n') {
        ++i;
      }
    } else {
      ++i;
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
