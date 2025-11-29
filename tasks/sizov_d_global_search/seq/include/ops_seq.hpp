#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sizov_d_global_search {

class SizovDGlobalSearchSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SizovDGlobalSearchSEQ(const InType &in);

 private:
  // Основная логика PPC
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  using Seed = std::pair<double, double>;

  // -------------------- Вспомогательные операции --------------------
  bool collect_initial_seeds(const InType &problem, std::vector<Seed> &seeds) const;
  void expand_seeds_around_best(const InType &problem, double uniq_eps, std::vector<Seed> &seeds) const;
  void initialize_state_from_seeds(const std::vector<Seed> &seeds);

  std::optional<double> evaluate_point(const InType &problem, double point, bool enforce_limit) const;

  struct SampleResult {
    double point{};
    double value{};
  };
  std::optional<SampleResult> seek_finite_sample(const InType &problem, double start, double direction) const;

  double dynamic_limit(double sample) const;
  double clamp_value(double value, bool *clipped) const;

  double estimate_m(double reliability) const;

  std::pair<double, bool> eval_interval_candidate(const InType &problem, double candidate, double left_point,
                                                  double right_point, double left_value, double right_value) const;

  double filter_candidate_value(double candidate, double left_value, double right_value, bool *clipped) const;

  bool try_insert_point(double point, double value, double insert_eps);

  // -------------------- Состояние алгоритма --------------------
  std::vector<double> points_;
  std::vector<double> values_;
  std::vector<bool> clipped_;

  double best_point_ = 0.0;
  double best_value_ = 0.0;
  int iterations_ = 0;
  bool converged_ = false;

  double value_ceiling_ = 1e6;
  double negative_ceiling_ = 1e6;
};

}  // namespace sizov_d_global_search
