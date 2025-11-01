#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "sizov_d_string_mismatch_count/common/include/common.hpp"
#include "sizov_d_string_mismatch_count/mpi/include/ops_mpi.hpp"
#include "sizov_d_string_mismatch_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_string_mismatch_count {

class SizovDRunFuncTestsStringMismatchCount : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string sanitized = test_param;
    std::replace(sanitized.begin(), sanitized.end(), ' ', '_');
    return sanitized;
  }

 protected:
  void SetUp() override {
    const std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_string_mismatch_count, "strings.txt");

    std::cerr << "[task] opening file: " << abs_path << "\n";

    std::ifstream file(abs_path);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string a;
    std::string b;

    if (rank == 0) {
      const std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_string_mismatch_count, "strings.txt");
      std::cerr << "[task] opening file: " << abs_path << "\n";

      std::ifstream file(abs_path);
      if (!file.is_open()) {
        throw std::runtime_error("Cannot open strings.txt");
      }

      std::getline(file, a);
      std::getline(file, b);
      file.close();

      TrimString(a);
      TrimString(b);
    }

    std::vector<char> buffer_a, buffer_b;

    // Передаём строки всем процессам (возможно лучше через MPI_Bcast)
    {
      int len_a = static_cast<int>(a.size());
      int len_b = static_cast<int>(b.size());

      MPI_Bcast(&len_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&len_b, 1, MPI_INT, 0, MPI_COMM_WORLD);

      buffer_a.resize(len_a);
      buffer_b.resize(len_b);

      if (rank == 0) {
        std::copy(a.begin(), a.end(), buffer_a.begin());
        std::copy(b.begin(), b.end(), buffer_b.begin());
      }

      MPI_Bcast(buffer_a.data(), len_a, MPI_CHAR, 0, MPI_COMM_WORLD);
      MPI_Bcast(buffer_b.data(), len_b, MPI_CHAR, 0, MPI_COMM_WORLD);

      a.assign(buffer_a.begin(), buffer_a.end());
      b.assign(buffer_b.begin(), buffer_b.end());
    }

    input_data_ = std::make_tuple(a, b);
    is_valid_ = !a.empty() && a.size() == b.size();

    int valid_flag = is_valid_ ? 1 : 0;
    MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!valid_flag) {
      if (rank == 0) {
        std::cerr << "[task] invalid input, skipping... a=" << a << ", b=" << b << "\n";
      }
      GTEST_SKIP();  // Теперь все ранки вызовут одинаково
    }

    expected_result_ = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) {
        ++expected_result_;
      }
    }

    std::cerr << "[task] setup complete: len=" << a.size() << ", mismatches=" << expected_result_ << "\n";

    std::cerr << "[task] setup complete: len=" << a.size() << ", mismatches=" << expected_result_ << "\n";
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    std::cerr << "[task] expected=" << expected_result_ << ", got=" << output_data << "\n";
    return output_data == expected_result_;
  }

 private:
  static void TrimString(std::string &s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return c == '\r' || c == '\n' || c == '\t'; }),
            s.end());
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
  }

  InType input_data_ = std::make_tuple(std::string{}, std::string{});
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

INSTANTIATE_TEST_SUITE_P(CompareFromFile, SizovDRunFuncTestsStringMismatchCount, kGtestValues, kTestName);

}  // namespace
}  // namespace sizov_d_string_mismatch_count
