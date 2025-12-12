#include "yurkin_counting_number/seq/include/ops_seq.hpp"

#include <cctype>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/include/util.hpp"
#include "yurkin_counting_number/common/include/common.hpp"

namespace yurkin_counting_number {

void SetTextForInput(const InType key, const std::string &text);
const std::string &GetTextForInput(const InType key);

static std::unordered_map<InType, std::string> g_text_store_;
static std::mutex g_text_store_mutex_;

void SetTextForInput(const InType key, const std::string &text) {
  std::lock_guard<std::mutex> lk(g_text_store_mutex_);
  g_text_store_[key] = text;
}

const std::string &GetTextForInput(const InType key) {
  std::lock_guard<std::mutex> lk(g_text_store_mutex_);
  static const std::string kEmpty;
  auto it = g_text_store_.find(key);
  if (it == g_text_store_.end()) {
    return kEmpty;
  }
  return it->second;
}

YurkinCountingNumberSEQ::YurkinCountingNumberSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YurkinCountingNumberSEQ::ValidationImpl() {
  return (GetInput() >= 0) && (GetOutput() == 0);
}

bool YurkinCountingNumberSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YurkinCountingNumberSEQ::RunImpl() {
  const std::string &text = GetTextForInput(GetInput());
  if (text.empty()) {
    return false;
  }

  int letters = 0;
  for (unsigned char c : text) {
    if (std::isalpha(c)) {
      ++letters;
    }
  }

  const int base_rep = 10000;
  int scale_down = 1 + (GetInput() / 100);
  int reps = std::max(1, base_rep / scale_down);

  volatile int dummy = 0;
  for (int r = 0; r < reps; ++r) {
    for (size_t i = 0; i < text.size(); ++i) {
      dummy += (text[i] & 0x1);
    }
  }
  (void)dummy;

  GetOutput() = letters;
  return true;
}

bool YurkinCountingNumberSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace yurkin_counting_number
