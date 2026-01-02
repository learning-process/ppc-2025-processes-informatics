#include "dorofeev_i_ccs_martrix_production/seq/include/ops_seq.hpp"

#include <map>

#include "dorofeev_i_ccs_martrix_production/common/include/common.hpp"

namespace dorofeev_i_ccs_martrix_production {

DorofeevICCSMatrixProductionSEQ::DorofeevICCSMatrixProductionSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DorofeevICCSMatrixProductionSEQ::ValidationImpl() {
  const auto &A = GetInput().first;
  const auto &B = GetInput().second;

  if (A.cols != B.rows) {
    return false;
  }

  if (A.col_ptr.size() != static_cast<size_t>(A.cols + 1)) {
    return false;
  }

  if (B.col_ptr.size() != static_cast<size_t>(B.cols + 1)) {
    return false;
  }

  return true;
}

bool DorofeevICCSMatrixProductionSEQ::PreProcessingImpl() {
  GetOutput().rows = GetInput().first.rows;
  GetOutput().cols = GetInput().second.cols;
  GetOutput().col_ptr.assign(GetOutput().cols + 1, 0);
  return true;
}

bool DorofeevICCSMatrixProductionSEQ::RunImpl() {
  const auto &A = GetInput().first;
  const auto &B = GetInput().second;
  auto &C = GetOutput();

  std::vector<std::map<int, double>> columns(C.cols);

  for (int j = 0; j < B.cols; j++) {
    for (int idxB = B.col_ptr[j]; idxB < B.col_ptr[j + 1]; idxB++) {
      int rowB = B.row_indices[idxB];
      double valB = B.values[idxB];

      for (int idxA = A.col_ptr[rowB]; idxA < A.col_ptr[rowB + 1]; idxA++) {
        int rowA = A.row_indices[idxA];
        double valA = A.values[idxA];
        columns[j][rowA] += valA * valB;
      }
    }
  }

  for (int j = 0; j < C.cols; j++) {
    C.col_ptr[j + 1] = C.col_ptr[j] + columns[j].size();
    for (const auto &[row, value] : columns[j]) {
      C.row_indices.push_back(row);
      C.values.push_back(value);
    }
  }

  return true;
}

bool DorofeevICCSMatrixProductionSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_ccs_martrix_production
