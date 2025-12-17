#include "akimov_i_star/seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "akimov_i_star/common/include/common.hpp"

namespace akimov_i_star {

namespace {

inline void Trim(std::string &s) {
  const char *ws = " \t\n\r\f\v";
  auto first = s.find_first_not_of(ws);
  if (first == std::string::npos) {
    s.clear();
    return;
  }
  auto last = s.find_last_not_of(ws);
  s = s.substr(first, last - first + 1);
}

std::vector<AkimovIStarSEQ::Op> ParseOpsFromString(const std::string &s) {
  std::vector<AkimovIStarSEQ::Op> res;
  std::istringstream ss(s);
  std::string line;
  while (std::getline(ss, line)) {
    Trim(line);
    if (line.empty()) {
      continue;
    }
    const std::string prefix = "send:";
    if (!line.starts_with(prefix)) {
      continue;
    }
    std::string rest = line.substr(prefix.size());
    size_t p1 = rest.find(':');
    if (p1 == std::string::npos) {
      continue;
    }
    size_t p2 = rest.find(':', p1 + 1);
    if (p2 == std::string::npos) {
      continue;
    }
    std::string srcs = rest.substr(0, p1);
    std::string dsts = rest.substr(p1 + 1, p2 - (p1 + 1));
    std::string msg = rest.substr(p2 + 1);
    Trim(srcs);
    Trim(dsts);
    Trim(msg);
    try {
      int src = std::stoi(srcs);
      int dst = std::stoi(dsts);
      res.push_back(AkimovIStarSEQ::Op{.src = src, .dst = dst, .msg = msg});
    } catch (...) {
      continue;
    }
  }
  return res;
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
  std::string s(input_buffer_.begin(), input_buffer_.end());
  ops_ = ParseOpsFromString(s);
  received_count_ = 0;
  return true;
}

bool AkimovIStarSEQ::RunImpl() {
  for (const auto &op : ops_) {
    if (op.dst == 0) {
      ++received_count_;
    }
  }
  GetOutput() = received_count_;
  return true;
}

bool AkimovIStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace akimov_i_star
