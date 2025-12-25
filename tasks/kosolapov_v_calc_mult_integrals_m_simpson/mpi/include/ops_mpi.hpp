#pragma once

#include <tuple>

#include "kosolapov_v_calc_mult_integrals_m_simpson/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kosolapov_v_calc_mult_integrals_m_simpson {

class KosolapovVCalcMultIntegralsMSimpsonMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KosolapovVCalcMultIntegralsMSimpsonMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  double Function1(double x, double y);
  double Function2(double x, double y);
  double Function3(double x, double y);
  double Function4(double x, double y);
  double CallFunction(int func_id, double x, double y);
  std::tuple<double, double, double, double> GetBounds(int func_id);
  double ComputePartialSimpsonIntegral(int func_id, int steps, double a, double b, double c, double d, int start_i,
                                       int end_i);
  double GetSimpsonWeight(int index, int n);
};

}  // namespace kosolapov_v_calc_mult_integrals_m_simpson
