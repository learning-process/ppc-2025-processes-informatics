#include <functional>
#include <utility>
#include <vector>

#include "frolova_s_mult_int_trapez/common/include/common.hpp"

namespace frolova_s_mult_int_trapez {

using InType = TrapezoidalIntegrationInput;
using OutType = double;

class FrolovaSMultIntTrapezSEQ : public BaseTask {
 public:
  explicit FrolovaSMultIntTrapezSEQ(const InType &in);

  TypeTask GetTypeTask() const override {
    return GetStaticTypeOfTask();
  }

  static TypeTask GetStaticTypeOfTask() {
    return TypeTask::kSequential;
  }

  FrolovaSMultIntTrapezSEQ(const FrolovaSMultIntTrapezSEQ &) = delete;
  FrolovaSMultIntTrapezSEQ &operator=(const FrolovaSMultIntTrapezSEQ &) = delete;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  unsigned int CalculationOfCoefficient(const std::vector<double> &point);
  void Recursive(std::vector<double> &point, unsigned int &definition, unsigned int divider, unsigned int variable);
  std::vector<double> GetPointFromNumber(unsigned int number);

  double result_{0.0};
  std::vector<std::pair<double, double>> limits_;
  std::vector<unsigned int> number_of_intervals_;
};

}  // namespace frolova_s_mult_int_trapez
