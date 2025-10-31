#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>  // для cerr
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"
#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"
#include "sizov_d_string_mismatch_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_string_mismatch_count {

inline int static GetMpiRankReallySafe() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    return 0;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

class SizovDRunFuncTestsStringMismatchCount : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    int rank = GetMpiRankReallySafe();
    std::string a, b;

    if (rank == 0) {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_string_mismatch_count, "strings.txt");

      std::cerr << "[test][rank 0] opening: " << abs_path << "\n";
      std::ifstream file(abs_path);
      if (!file.is_open()) {
        std::cerr << "[test][rank 0] FAILED to open file\n";
        a = "";
        b = "";
      } else {
        std::getline(file, a);
        std::getline(file, b);
        file.close();
      }
    }

    int len_a = static_cast<int>(a.size());
    int len_b = static_cast<int>(b.size());
    MPI_Bcast(&len_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&len_b, 1, MPI_INT, 0, MPI_COMM_WORLD);

    a.resize(len_a);
    b.resize(len_b);
    if (len_a > 0) {
      MPI_Bcast(a.data(), len_a, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    if (len_b > 0) {
      MPI_Bcast(b.data(), len_b, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    input_data_ = std::make_tuple(a, b);
    is_valid_ = !a.empty() && a.size() == b.size();

    expected_result_ = 0;
    if (is_valid_) {
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
          ++expected_result_;
        }
      }
    }

    std::cerr << "[test][rank " << rank << "] setup complete, len_a=" << len_a << " len_b=" << len_b << "\n";
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (!is_valid_) {
      return true;
    }

    auto safe_get_rank = []() -> std::optional<int> {
#ifdef _WIN32
      char *buf = nullptr;
      size_t len = 0;
      if (_dupenv_s(&buf, &len, "OMPI_COMM_WORLD_RANK") == 0 && buf) {
        char *end_ptr = nullptr;
        size_t val = std::strtol(buf, &end_ptr, 10);
        free(buf);
        if (end_ptr != buf && *end_ptr == '\0') {
          return static_cast<int>(val);
        }
      }
      return std::nullopt;
#else
      return std::nullopt;
#endif
    };

    const int rank = safe_get_rank().value_or(0);

    if (rank != 0) {
      return true;
    }

    return output_data == expected_result_;
  }

 private:
  InType input_data_;
  OutType expected_result_ = 0;
  bool is_valid_ = true;
};

namespace {

TEST_P(SizovDRunFuncTestsStringMismatchCount, CompareStringsFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {"default"};

const auto kTaskList = std::tuple_cat(ppc::util::AddFuncTask<SizovDStringMismatchCountMPI, InType>(
                                          kTestParam, PPC_SETTINGS_sizov_d_string_mismatch_count),
                                      ppc::util::AddFuncTask<SizovDStringMismatchCountSEQ, InType>(
                                          kTestParam, PPC_SETTINGS_sizov_d_string_mismatch_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTaskList);
const auto kTestName = SizovDRunFuncTestsStringMismatchCount::PrintFuncTestName<SizovDRunFuncTestsStringMismatchCount>;

INSTANTIATE_TEST_SUITE_P(CompareFromFile, SizovDRunFuncTestsStringMismatchCount, kGtestValues,
                         kTestName);  // NOLINT

}  // namespace
}  // namespace sizov_d_string_mismatch_count
