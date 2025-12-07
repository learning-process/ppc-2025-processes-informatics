#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_global_search {

namespace {

struct ReferenceResult {
  double argmin = 0.0;
  double value = 0.0;
};

enum class TokenType : std::uint8_t {
  kEnd,
  kNumber,
  kIdentifier,
  kPlus,
  kMinus,
  kStar,
  kSlash,
  kCaret,
  kLParen,
  kRParen
};

struct Token {
  TokenType type = TokenType::kEnd;
  std::string text;
  double number_value = 0.0;
};

class Lexer {
 public:
  Lexer(std::string expression, std::filesystem::path path)
      : expression_(std::move(expression)), path_(std::move(path)) {}

  Token Next() {
    SkipWhitespace();
    if (pos_ >= expression_.size()) {
      Token t;
      t.type = TokenType::kEnd;
      return t;
    }

    const char ch = expression_[pos_];

    if ((std::isdigit(static_cast<unsigned char>(ch)) != 0) || ch == '.') {
      return ParseNumber();
    }

    if (IsIdentifierStart(ch)) {
      return ParseIdentifier();
    }

    ++pos_;
    Token t;
    switch (ch) {
      case '+':
        t.type = TokenType::kPlus;
        t.text = "+";
        return t;
      case '-':
        t.type = TokenType::kMinus;
        t.text = "-";
        return t;
      case '*':
        t.type = TokenType::kStar;
        t.text = "*";
        return t;
      case '/':
        t.type = TokenType::kSlash;
        t.text = "/";
        return t;
      case '^':
        t.type = TokenType::kCaret;
        t.text = "^";
        return t;
      case '(':
        t.type = TokenType::kLParen;
        t.text = "(";
        return t;
      case ')':
        t.type = TokenType::kRParen;
        t.text = ")";
        return t;
      default:
        break;
    }

    throw std::runtime_error("Unexpected character '" + std::string(1, ch) + "' at position " +
                             std::to_string(pos_ - 1) + " in expression " + path_.string());
  }

 private:
  void SkipWhitespace() {
    while (pos_ < expression_.size() && (std::isspace(static_cast<unsigned char>(expression_[pos_])) != 0)) {
      ++pos_;
    }
  }

  Token ParseNumber() {
    const char *begin = expression_.c_str() + pos_;
    char *end = nullptr;
    const double value = std::strtod(begin, &end);
    if (begin == end) {
      throw std::runtime_error("Invalid number in " + path_.string());
    }
    pos_ = static_cast<std::size_t>(end - expression_.c_str());

    Token t;
    t.type = TokenType::kNumber;
    t.number_value = value;
    return t;
  }

  Token ParseIdentifier() {
    const std::size_t start = pos_;
    while (pos_ < expression_.size() && IsIdentifierChar(expression_[pos_])) {
      ++pos_;
    }
    Token token;
    token.type = TokenType::kIdentifier;
    token.text = expression_.substr(start, pos_ - start);
    return token;
  }

  static bool IsIdentifierStart(char ch) {
    return (std::isalpha(static_cast<unsigned char>(ch)) != 0) || ch == '_';
  }

  static bool IsIdentifierChar(char ch) {
    return (std::isalnum(static_cast<unsigned char>(ch)) != 0) || ch == '_';
  }

  std::string expression_;
  std::filesystem::path path_;
  std::size_t pos_ = 0;
};

class Expr {
 public:
  virtual ~Expr() = default;
  [[nodiscard]] virtual double Eval(double x) const = 0;
};

using ExprPtr = std::shared_ptr<Expr>;

class ConstantExpr : public Expr {
 public:
  explicit ConstantExpr(double v) : v_(v) {}
  [[nodiscard]] double Eval(double /*x*/) const override {
    return v_;
  }

 private:
  double v_;
};

class VariableExpr : public Expr {
 public:
  VariableExpr() = default;
  [[nodiscard]] double Eval(double x) const override {
    return x;
  }
};

class UnaryExpr : public Expr {
 public:
  UnaryExpr(char op, ExprPtr child) : op_(op), child_(std::move(child)) {}
  [[nodiscard]] double Eval(double x) const override {
    const double v = child_->Eval(x);
    if (op_ == '-') {
      return -v;
    }
    return v;
  }

 private:
  char op_;
  ExprPtr child_;
};

class BinaryExpr : public Expr {
 public:
  BinaryExpr(char op, ExprPtr l, ExprPtr r) : op_(op), left_(std::move(l)), right_(std::move(r)) {}

  [[nodiscard]] double Eval(double x) const override {
    const double a = left_->Eval(x);
    const double b = right_->Eval(x);
    switch (op_) {
      case '+':
        return a + b;
      case '-':
        return a - b;
      case '*':
        return a * b;
      case '/':
        return a / b;
      case '^':
        return std::pow(a, b);
      default:
        return std::numeric_limits<double>::quiet_NaN();
    }
  }

 private:
  char op_;
  ExprPtr left_;
  ExprPtr right_;
};

class FunctionCallExpr : public Expr {
 public:
  using FuncPtr = double (*)(double);

  FunctionCallExpr(FuncPtr f, ExprPtr arg) : func_(f), arg_(std::move(arg)) {}

  [[nodiscard]] double Eval(double x) const override {
    return func_(arg_->Eval(x));
  }

 private:
  FuncPtr func_;
  ExprPtr arg_;
};

class Parser {
 public:
  Parser(Lexer lex, std::filesystem::path path) : lexer_(std::move(lex)), path_(std::move(path)) {
    Advance();
  }

  ExprPtr Parse() {
    ExprPtr expr = ParseExpression();
    Expect(TokenType::kEnd);
    return expr;
  }

 private:
  struct OperatorEntry {
    enum class Kind : std::uint8_t { kPlus, kMinus, kMul, kDiv, kPow, kUnaryMinus, kLParen, kFunc };

    Kind kind = Kind::kPlus;
    FunctionCallExpr::FuncPtr func = nullptr;
  };

  void Advance() {
    current_ = lexer_.Next();
  }

  [[nodiscard]] bool Check(TokenType t) const {
    return current_.type == t;
  }

  void Expect(TokenType t) {
    if (!Check(t)) {
      throw std::runtime_error("Unexpected token in " + path_.string());
    }
    Advance();
  }

  static int Precedence(const OperatorEntry &op) {
    switch (op.kind) {
      case OperatorEntry::Kind::kPlus:
      case OperatorEntry::Kind::kMinus:
        return 1;
      case OperatorEntry::Kind::kMul:
      case OperatorEntry::Kind::kDiv:
        return 2;
      case OperatorEntry::Kind::kPow:
        return 3;
      case OperatorEntry::Kind::kUnaryMinus:
      case OperatorEntry::Kind::kFunc:
        return 4;
      case OperatorEntry::Kind::kLParen:
      default:
        return 0;
    }
  }

  static bool IsRightAssociative(const OperatorEntry &op) {
    return op.kind == OperatorEntry::Kind::kPow || op.kind == OperatorEntry::Kind::kUnaryMinus;
  }

  static void ApplyOperator(std::vector<ExprPtr> &values, const OperatorEntry &op, const std::filesystem::path &path) {
    switch (op.kind) {
      case OperatorEntry::Kind::kUnaryMinus: {
        if (values.empty()) {
          throw std::runtime_error("Missing operand for unary '-' in " + path.string());
        }
        auto a = values.back();
        values.pop_back();
        values.push_back(std::make_shared<UnaryExpr>('-', a));
        return;
      }

      case OperatorEntry::Kind::kFunc: {
        if (values.empty()) {
          throw std::runtime_error("Missing operand for function call in " + path.string());
        }
        auto arg = values.back();
        values.pop_back();
        values.push_back(std::make_shared<FunctionCallExpr>(op.func, arg));
        return;
      }

      case OperatorEntry::Kind::kPlus:
      case OperatorEntry::Kind::kMinus:
      case OperatorEntry::Kind::kMul:
      case OperatorEntry::Kind::kDiv:
      case OperatorEntry::Kind::kPow: {
        if (values.size() < 2U) {
          throw std::runtime_error("Missing operands for binary operator in " + path.string());
        }
        auto rhs = values.back();
        values.pop_back();
        auto lhs = values.back();
        values.pop_back();

        char op_char = '?';
        if (op.kind == OperatorEntry::Kind::kPlus) {
          op_char = '+';
        } else if (op.kind == OperatorEntry::Kind::kMinus) {
          op_char = '-';
        } else if (op.kind == OperatorEntry::Kind::kMul) {
          op_char = '*';
        } else if (op.kind == OperatorEntry::Kind::kDiv) {
          op_char = '/';
        } else if (op.kind == OperatorEntry::Kind::kPow) {
          op_char = '^';
        }

        values.push_back(std::make_shared<BinaryExpr>(op_char, lhs, rhs));
        return;
      }

      case OperatorEntry::Kind::kLParen:
      default:
        return;
    }
  }

  static FunctionCallExpr::FuncPtr ResolveFunction(const std::string &name) {
    static const std::unordered_map<std::string, double (*)(double)> kMap = {
        {"sin", std::sin}, {"cos", std::cos},   {"tan", std::tan}, {"exp", std::exp},
        {"log", std::log}, {"sqrt", std::sqrt}, {"abs", std::fabs}};
    const auto it = kMap.find(name);
    if (it == kMap.end()) {
      throw std::runtime_error("Unknown function '" + name + "'");
    }
    return it->second;
  }

  static void PushUnaryMinus(std::vector<OperatorEntry> &ops) {
    OperatorEntry op;
    op.kind = OperatorEntry::Kind::kUnaryMinus;
    op.func = nullptr;
    ops.push_back(op);
  }

  static void PushLParen(std::vector<OperatorEntry> &ops) {
    OperatorEntry op;
    op.kind = OperatorEntry::Kind::kLParen;
    op.func = nullptr;
    ops.push_back(op);
  }

  static void PushFunction(std::vector<OperatorEntry> &ops, const std::string &id) {
    OperatorEntry func_op;
    func_op.kind = OperatorEntry::Kind::kFunc;
    func_op.func = ResolveFunction(id);
    ops.push_back(func_op);

    PushLParen(ops);
  }

  void PushBinaryOperator(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, OperatorEntry::Kind kind) {
    OperatorEntry new_op;
    new_op.kind = kind;
    new_op.func = nullptr;

    while (!ops.empty() && ops.back().kind != OperatorEntry::Kind::kLParen) {
      const int top_prec = Precedence(ops.back());
      const int new_prec = Precedence(new_op);

      if (top_prec > new_prec || (!IsRightAssociative(new_op) && top_prec == new_prec)) {
        ApplyOperator(values, ops.back(), path_);
        ops.pop_back();
      } else {
        break;
      }
    }

    ops.push_back(new_op);
  }

  void HandleIdentifierToken(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, bool &expect_value) {
    const std::string id = current_.text;
    Advance();

    if (Check(TokenType::kLParen)) {
      PushFunction(ops, id);
      Advance();
      expect_value = true;
      return;
    }

    if (id == "x") {
      values.push_back(std::make_shared<VariableExpr>());
      expect_value = false;
      return;
    }

    if (id == "pi") {
      values.push_back(std::make_shared<ConstantExpr>(std::acos(-1.0)));
      expect_value = false;
      return;
    }

    throw std::runtime_error("Unknown identifier '" + id + "'");
  }

  void HandleExpectValue(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, bool &expect_value) {
    if (Check(TokenType::kNumber)) {
      const double v = current_.number_value;
      Advance();
      values.push_back(std::make_shared<ConstantExpr>(v));
      expect_value = false;
      return;
    }

    if (Check(TokenType::kIdentifier)) {
      HandleIdentifierToken(values, ops, expect_value);
      return;
    }

    if (Check(TokenType::kLParen)) {
      PushLParen(ops);
      Advance();
      expect_value = true;
      return;
    }

    if (Check(TokenType::kPlus) || Check(TokenType::kMinus)) {
      if (Check(TokenType::kMinus)) {
        PushUnaryMinus(ops);
      }
      Advance();
      expect_value = true;
      return;
    }

    throw std::runtime_error("Expected value in " + path_.string());
  }

  void HandleRParen(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops) {
    while (!ops.empty() && ops.back().kind != OperatorEntry::Kind::kLParen) {
      ApplyOperator(values, ops.back(), path_);
      ops.pop_back();
    }
    if (ops.empty()) {
      throw std::runtime_error("Mismatched parentheses in " + path_.string());
    }
    ops.pop_back();

    if (!ops.empty() && ops.back().kind == OperatorEntry::Kind::kFunc) {
      ApplyOperator(values, ops.back(), path_);
      ops.pop_back();
    }
  }

  bool HandleOperatorOrEnd(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, bool &expect_value) {
    if (Check(TokenType::kPlus) || Check(TokenType::kMinus) || Check(TokenType::kStar) || Check(TokenType::kSlash) ||
        Check(TokenType::kCaret)) {
      OperatorEntry::Kind kind = OperatorEntry::Kind::kPlus;

      if (Check(TokenType::kPlus)) {
        kind = OperatorEntry::Kind::kPlus;
      } else if (Check(TokenType::kMinus)) {
        kind = OperatorEntry::Kind::kMinus;
      } else if (Check(TokenType::kStar)) {
        kind = OperatorEntry::Kind::kMul;
      } else if (Check(TokenType::kSlash)) {
        kind = OperatorEntry::Kind::kDiv;
      } else {
        kind = OperatorEntry::Kind::kPow;
      }

      PushBinaryOperator(values, ops, kind);
      Advance();
      expect_value = true;
      return true;
    }

    if (Check(TokenType::kRParen)) {
      HandleRParen(values, ops);
      Advance();
      return true;
    }

    if (Check(TokenType::kEnd)) {
      return false;
    }

    throw std::runtime_error("Unexpected token in " + path_.string());
  }

  void FinalizeExpression(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops) {
    while (!ops.empty()) {
      if (ops.back().kind == OperatorEntry::Kind::kLParen) {
        throw std::runtime_error("Mismatched parentheses in " + path_.string());
      }
      ApplyOperator(values, ops.back(), path_);
      ops.pop_back();
    }

    if (values.size() != 1U) {
      throw std::runtime_error("Invalid expression in " + path_.string());
    }
  }

  ExprPtr ParseExpression() {
    std::vector<ExprPtr> values;
    std::vector<OperatorEntry> ops;
    bool expect_value = true;

    while (true) {
      if (expect_value) {
        HandleExpectValue(values, ops, expect_value);
      } else {
        if (!HandleOperatorOrEnd(values, ops, expect_value)) {
          break;
        }
      }
    }

    FinalizeExpression(values, ops);
    return values.back();
  }

  Lexer lexer_;
  Token current_;
  std::filesystem::path path_;
};

Function BuildFunction(const nlohmann::json &json, const std::filesystem::path &path) {
  if (!json.contains("expression")) {
    throw std::runtime_error("Missing expression field");
  }

  const std::string expr = json.at("expression").get<std::string>();

  Lexer lex(expr, path);
  Parser parser(std::move(lex), path);
  ExprPtr ast = parser.Parse();

  return Function{[ast](double x) { return ast->Eval(x); }};
}

[[nodiscard]] double ComputeSafeStep(double requested_step, double span) {
  if (span <= 0.0) {
    return 1e-4;
  }
  if (requested_step > 0.0 && requested_step < span) {
    return requested_step;
  }
  const double by_points = span / 1000.0;
  return std::max(by_points, 1e-4);
}

[[nodiscard]] bool IsIncrementAcceptable(double value, double last_value, double dx, double value_threshold,
                                         double slope_threshold) {
  if (!std::isfinite(last_value)) {
    return true;
  }
  const double diff = std::abs(value - last_value);
  const double slope = (dx != 0.0) ? diff / std::abs(dx) : std::numeric_limits<double>::infinity();

  if (diff > value_threshold) {
    return false;
  }
  if (diff > 1e5 * std::abs(last_value)) {
    return false;
  }
  if (slope > slope_threshold) {
    return false;
  }
  return true;
}

[[nodiscard]] bool ShouldAcceptRightEndpoint(double vr, double last_value, double value_threshold) {
  if (!std::isfinite(vr)) {
    return false;
  }
  if (!std::isfinite(last_value)) {
    return true;
  }

  const double diff = std::abs(vr - last_value);

  if (diff > value_threshold) {
    return false;
  }

  if (diff > 1e5 * std::abs(last_value)) {
    return false;
  }

  return true;
}

ReferenceResult BruteForceReference(const Problem &p, double step) {
  const double span = p.right - p.left;
  const double safe_step = ComputeSafeStep(step, span);

  ReferenceResult r;
  r.argmin = p.left;
  r.value = std::numeric_limits<double>::infinity();

  double x = p.left;
  constexpr double kValueThreshold = 1e3;
  constexpr double kSlopeThreshold = 1e3;

  double last_eval_value = std::numeric_limits<double>::quiet_NaN();
  double last_eval_x = std::numeric_limits<double>::quiet_NaN();

  while (x <= p.right + 1e-12) {
    const double v = p.func(x);
    bool accept = std::isfinite(v);

    if (accept && std::isfinite(last_eval_value)) {
      const double dx = x - last_eval_x;
      if (!IsIncrementAcceptable(v, last_eval_value, dx, kValueThreshold, kSlopeThreshold)) {
        accept = false;
      }
    }

    if (accept && v < r.value) {
      r.value = v;
      r.argmin = x;
    }

    if (std::isfinite(v)) {
      last_eval_value = v;
      last_eval_x = x;
    }

    x += safe_step;
  }

  const double vr = p.func(p.right);
  if (ShouldAcceptRightEndpoint(vr, last_eval_value, kValueThreshold) && vr < r.value) {
    r.value = vr;
    r.argmin = p.right;
  }

  return r;
}

std::vector<TestCase> LoadTestCasesFromData() {
  namespace fs = std::filesystem;
  const fs::path json_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_global_search, "tests.json");

  std::ifstream file(json_path);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open tests.json");
  }

  nlohmann::json data;
  file >> data;

  if (!data.is_array()) {
    throw std::runtime_error("tests.json must contain array");
  }

  std::vector<TestCase> cases;
  cases.reserve(data.size());

  for (auto &item : data) {
    TestCase t;
    t.name = item.at("name").get<std::string>();

    const auto &pj = item.at("problem");

    Problem p{};
    p.left = pj.at("left");
    p.right = pj.at("right");

    p.accuracy = 1e-4;
    p.reliability = 2.5;
    p.max_iterations = 5000;

    p.func = BuildFunction(item.at("function"), json_path);

    t.problem = p;

    t.brute_force_step = 1e-5;
    t.tolerance = 0.002;

    cases.push_back(std::move(t));
  }

  return cases;
}

using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

std::vector<FuncParam> BuildTestTasks(const std::vector<TestType> &tests) {
  std::vector<FuncParam> tasks;
  tasks.reserve(tests.size() * 2U);

  const std::string mpi_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchMPI>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchMPI::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  const std::string seq_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchSEQ>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchSEQ::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  for (const auto &t : tests) {
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchMPI, InType>, mpi_name, t);
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchSEQ, InType>, seq_name, t);
  }

  return tasks;
}

}  // namespace

class SizovDGlobalSearchFunctionalTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_case) {
    return test_case.name;
  }

 protected:
  void SetUp() override {
    test_case_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = test_case_.problem;

    reference_ = BruteForceReference(input_, test_case_.brute_force_step);
  }

  bool CheckTestOutputData(OutType &o) final {
    const double dx = std::abs(o.argmin - reference_.argmin);
    const double dv = std::abs(o.value - reference_.value);

    const double dx_tol = test_case_.tolerance;
    const double dv_tol = 5.0 * test_case_.tolerance;

    const bool good_numerically = (dx <= dx_tol && dv <= dv_tol);

    bool cross_match = false;
    if (dv <= dv_tol) {
      const auto &problem = input_;
      const double ref_at_result = problem.func(o.argmin);
      const double result_at_ref = problem.func(reference_.argmin);
      cross_match =
          (std::abs(ref_at_result - reference_.value) <= dv_tol) && (std::abs(result_at_ref - o.value) <= dv_tol);
    }

    return good_numerically || cross_match;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  TestType test_case_;
  InType input_;
  ReferenceResult reference_{};
};

namespace {

TEST_P(SizovDGlobalSearchFunctionalTests, FindsGlobalMinimum) {
  ExecuteTest(GetParam());
}

const auto kTests = LoadTestCasesFromData();
const auto kTasks = BuildTestTasks(kTests);
const auto kGtestValues = ::testing::ValuesIn(kTasks);

const auto kTestName = SizovDGlobalSearchFunctionalTests::PrintFuncTestName<SizovDGlobalSearchFunctionalTests>;

INSTANTIATE_TEST_SUITE_P(GlobalSearch, SizovDGlobalSearchFunctionalTests, kGtestValues, kTestName);

}  // namespace

}  // namespace sizov_d_global_search
