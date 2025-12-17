#include "akimov_i_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "akimov_i_star/common/include/common.hpp"

namespace akimov_i_star {

namespace {

std::string ToString(const InType &buf) {
  return std::string(buf.begin(), buf.end());
}

void FromString(const std::string &s, InType &out) {
  out.assign(s.begin(), s.end());
}

static void Trim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

std::vector<AkimovIStarMPI::Op> ParseOpsFromString(const std::string &s) {
  std::vector<AkimovIStarMPI::Op> res;
  std::istringstream ss(s);
  std::string line;
  while (std::getline(ss, line)) {
    Trim(line);
    if (line.empty()) {
      continue;
    }
    const std::string prefix = "send:";
    if (line.rfind(prefix, 0) != 0) {
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
      res.push_back(AkimovIStarMPI::Op{src, dst, msg});
    } catch (...) {
      continue;
    }
  }
  return res;
}

}  // namespace

AkimovIStarMPI::AkimovIStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool AkimovIStarMPI::ValidationImpl() {
  int rank = 0;
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    return true;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool AkimovIStarMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  input_buffer_.clear();
  ops_.clear();
  received_count_ = 0;

  std::string raw;
  if (rank == 0) {
    raw = ToString(GetInput());
  }

  int len = static_cast<int>(raw.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (len < 0) {
    len = 0;
  }
  std::string buf;
  buf.resize(len);
  MPI_Bcast(len ? &buf[0] : nullptr, len, MPI_CHAR, 0, MPI_COMM_WORLD);

  input_buffer_.clear();
  if (len > 0) {
    input_buffer_.assign(buf.begin(), buf.end());
  }

  std::string parsed = ToString(input_buffer_);
  ops_ = ParseOpsFromString(parsed);

  return true;
}

bool AkimovIStarMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size == 1) {
    int cnt = 0;
    for (const auto &op : ops_) {
      if (op.dst == 0) {
        ++cnt;
      }
    }
    received_count_ = cnt;
    GetOutput() = received_count_;
    return true;
  }

  int local_send_count = 0;
  int local_expected_recv = 0;
  for (const auto &op : ops_) {
    if (op.src == rank) {
      ++local_send_count;
    }
    if (op.dst == rank) {
      ++local_expected_recv;
    }
  }

  int total_sends = 0;
  MPI_Allreduce(&local_send_count, &total_sends, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  const int center = 0;

  if (rank != center) {
    for (const auto &op : ops_) {
      if (op.src != rank) {
        continue;
      }
      int header[2];
      header[0] = op.dst;
      header[1] = static_cast<int>(op.msg.size());
      MPI_Send(header, 2, MPI_INT, center, 0, MPI_COMM_WORLD);
      if (header[1] > 0) {
        MPI_Send(op.msg.data(), header[1], MPI_CHAR, center, 0, MPI_COMM_WORLD);
      }
    }
    int recvd = 0;
    for (int i = 0; i < local_expected_recv; ++i) {
      int header[2] = {0, 0};
      MPI_Recv(header, 2, MPI_INT, center, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int payload_len = header[1];
      std::string payload;
      payload.resize(payload_len);
      if (payload_len > 0) {
        MPI_Recv(&payload[0], payload_len, MPI_CHAR, center, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      ++recvd;
    }
    received_count_ = recvd;
  } else {
    for (const auto &op : ops_) {
      if (op.src != center) {
        continue;
      }
      if (op.dst == center) {
        ++received_count_;
      } else {
        int header[2];
        header[0] = op.src;
        header[1] = static_cast<int>(op.msg.size());
        MPI_Send(header, 2, MPI_INT, op.dst, 0, MPI_COMM_WORLD);
        if (header[1] > 0) {
          MPI_Send(op.msg.data(), header[1], MPI_CHAR, op.dst, 0, MPI_COMM_WORLD);
        }
      }
    }

    int center_local_sends = 0;
    for (const auto &op : ops_) {
      if (op.src == center) {
        ++center_local_sends;
      }
    }
    int recv_from_others = total_sends - center_local_sends;
    for (int i = 0; i < recv_from_others; ++i) {
      int header[2] = {0, 0};
      MPI_Recv(header, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int dst = header[0];
      int payload_len = header[1];
      std::string payload;
      payload.resize(payload_len);
      if (payload_len > 0) {
        MPI_Recv(&payload[0], payload_len, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (dst == center) {
        ++received_count_;
      } else {
        int fwd_header[2];
        fwd_header[0] = 0;
        fwd_header[1] = payload_len;
        MPI_Send(fwd_header, 2, MPI_INT, dst, 0, MPI_COMM_WORLD);
        if (payload_len > 0) {
          MPI_Send(payload.data(), payload_len, MPI_CHAR, dst, 0, MPI_COMM_WORLD);
        }
      }
    }
  }

  GetOutput() = received_count_;
  return true;
}

bool AkimovIStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace akimov_i_star
