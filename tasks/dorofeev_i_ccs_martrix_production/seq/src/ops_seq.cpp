#include "dorofeev_i_ccs_martrix_production/seq/include/ops_seq.hpp"

#include <map>
#include <vector>

#include "dorofeev_i_ccs_martrix_production/common/include/common.hpp"

namespace dorofeev_i_ccs_martrix_production {

DorofeevICCSMatrixProductionSEQ::DorofeevICCSMatrixProductionSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DorofeevICCSMatrixProductionSEQ::ValidationImpl() {
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;

  if (a.cols != b.rows) {
    return false;
  }

  if (static_cast<int>(a.col_ptr.size()) != a.cols + 1) {
    return false;
  }

  if (static_cast<int>(b.col_ptr.size()) != b.cols + 1) {
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
  const auto &a = GetInput().first;
  const auto &b = GetInput().second;
  auto &c = GetOutput();

  std::vector<std::map<int, double>> columns(c.cols);

  for (int j = 0; j < b.cols; j++) {
    for (int idx_b = b.col_ptr[j]; idx_b < b.col_ptr[j + 1]; idx_b++) {
      int row_b = b.row_indices[idx_b];
      double val_b = b.values[idx_b];

      for (int idx_a = a.col_ptr[row_b]; idx_a < a.col_ptr[row_b + 1]; idx_a++) {
        int row_a = a.row_indices[idx_a];
        double val_a = a.values[idx_a];
        columns[j][row_a] += val_a * val_b;
      }
    }
  }

  for (int j = 0; j < c.cols; j++) {
    c.col_ptr[j + 1] = c.col_ptr[j] + static_cast<int>(columns[j].size());
    for (const auto &[row, value] : columns[j]) {
      c.row_indices.push_back(row);
      c.values.push_back(value);
    }
  }

  return true;
}

bool DorofeevICCSMatrixProductionSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dorofeev_i_ccs_martrix_production
