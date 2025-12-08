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
    if (ch == '+') {
      t.type = TokenType::kPlus;
      t.text = "+";
      return t;
    }
    if (ch == '-') {
      t.type = TokenType::kMinus;
      t.text = "-";
      return t;
    }
    if (ch == '*') {
      t.type = TokenType::kStar;
      t.text = "*";
      return t;
    }
    if (ch == '/') {
      t.type = TokenType::kSlash;
      t.text = "/";
      return t;
    }
    if (ch == '^') {
      t.type = TokenType::kCaret;
      t.text = "^";
      return t;
    }
    if (ch == '(') {
      t.type = TokenType::kLParen;
      t.text = "(";
      return t;
    }
    if (ch == ')') {
      t.type = TokenType::kRParen;
      t.text = ")";
      return t;
    }
    throw std::runtime_error("Unexpected character");
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
      throw std::runtime_error("Invalid number");
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
  virtual double Eval(double x) const = 0;
};

using ExprPtr = std::unique_ptr<Expr>;

class ConstantExpr : public Expr {
 public:
  explicit ConstantExpr(double v) : v_(v) {}
  double Eval(double) const override {
    return v_;
  }

 private:
  double v_;
};

class VariableExpr : public Expr {
 public:
  double Eval(double x) const override {
    return x;
  }
};

class UnaryExpr : public Expr {
 public:
  UnaryExpr(char op, ExprPtr child) : op_(op), child_(std::move(child)) {}
  double Eval(double x) const override {
    double v = child_->Eval(x);
    return (op_ == '-') ? -v : v;
  }

 private:
  char op_;
  ExprPtr child_;
};

class BinaryExpr : public Expr {
 public:
  BinaryExpr(char op, ExprPtr l, ExprPtr r) : op_(op), left_(std::move(l)), right_(std::move(r)) {}
  double Eval(double x) const override {
    double a = left_->Eval(x);
    double b = right_->Eval(x);
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
    }
    return std::numeric_limits<double>::quiet_NaN();
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
  double Eval(double x) const override {
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
    Kind kind;
    FunctionCallExpr::FuncPtr func = nullptr;
  };

  void Advance() {
    current_ = lexer_.Next();
  }

  bool Check(TokenType t) const {
    return current_.type == t;
  }

  void Expect(TokenType t) {
    if (!Check(t)) {
      throw std::runtime_error("Unexpected token");
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
        return 0;
    }
    return 0;
  }

  static bool IsRightAssociative(const OperatorEntry &op) {
    return op.kind == OperatorEntry::Kind::kPow || op.kind == OperatorEntry::Kind::kUnaryMinus;
  }

  static void ApplyOperator(std::vector<ExprPtr> &values, const OperatorEntry &op) {
    switch (op.kind) {
      case OperatorEntry::Kind::kUnaryMinus: {
        auto a = std::move(values.back());
        values.pop_back();
        values.push_back(std::make_unique<UnaryExpr>('-', std::move(a)));
        return;
      }

      case OperatorEntry::Kind::kFunc: {
        auto arg = std::move(values.back());
        values.pop_back();
        values.push_back(std::make_unique<FunctionCallExpr>(op.func, std::move(arg)));
        return;
      }

      case OperatorEntry::Kind::kPlus:
      case OperatorEntry::Kind::kMinus:
      case OperatorEntry::Kind::kMul:
      case OperatorEntry::Kind::kDiv:
      case OperatorEntry::Kind::kPow: {
        auto rhs = std::move(values.back());
        values.pop_back();
        auto lhs = std::move(values.back());
        values.pop_back();

        char c = '?';
        if (op.kind == OperatorEntry::Kind::kPlus) {
          c = '+';
        } else if (op.kind == OperatorEntry::Kind::kMinus) {
          c = '-';
        } else if (op.kind == OperatorEntry::Kind::kMul) {
          c = '*';
        } else if (op.kind == OperatorEntry::Kind::kDiv) {
          c = '/';
        } else if (op.kind == OperatorEntry::Kind::kPow) {
          c = '^';
        }

        values.push_back(std::make_unique<BinaryExpr>(c, std::move(lhs), std::move(rhs)));
        return;
      }

      case OperatorEntry::Kind::kLParen:
        return;
    }
  }

  static FunctionCallExpr::FuncPtr ResolveFunction(const std::string &name) {
    if (name == "sin") {
      return std::sin;
    }
    if (name == "cos") {
      return std::cos;
    }
    if (name == "tan") {
      return std::tan;
    }
    if (name == "exp") {
      return std::exp;
    }
    if (name == "log") {
      return std::log;
    }
    if (name == "sqrt") {
      return std::sqrt;
    }
    if (name == "abs") {
      return static_cast<double (*)(double)>(std::fabs);
    }
    throw std::runtime_error("Unknown function");
  }

  void PushUnaryMinus(std::vector<OperatorEntry> &ops) {
    ops.push_back({OperatorEntry::Kind::kUnaryMinus, nullptr});
  }

  void PushLParen(std::vector<OperatorEntry> &ops) {
    ops.push_back({OperatorEntry::Kind::kLParen, nullptr});
  }

  void PushFunction(std::vector<OperatorEntry> &ops, const std::string &id) {
    ops.push_back({OperatorEntry::Kind::kFunc, ResolveFunction(id)});
    PushLParen(ops);
  }

  void PushBinaryOperator(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, OperatorEntry::Kind kind) {
    OperatorEntry new_op{kind, nullptr};
    while (!ops.empty() && ops.back().kind != OperatorEntry::Kind::kLParen) {
      int top = Precedence(ops.back());
      int now = Precedence(new_op);
      if (top > now || (!IsRightAssociative(new_op) && top == now)) {
        ApplyOperator(values, ops.back());
        ops.pop_back();
      } else {
        break;
      }
    }
    ops.push_back(new_op);
  }

  void HandleIdentifierToken(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, bool &expect_value) {
    std::string id = current_.text;
    Advance();
    if (Check(TokenType::kLParen)) {
      PushFunction(ops, id);
      Advance();
      expect_value = true;
      return;
    }
    if (id == "x") {
      values.push_back(std::make_unique<VariableExpr>());
      expect_value = false;
      return;
    }
    if (id == "pi") {
      values.push_back(std::make_unique<ConstantExpr>(std::acos(-1.0)));
      expect_value = false;
      return;
    }
    throw std::runtime_error("Unknown identifier");
  }

  void HandleExpectValue(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, bool &expect_value) {
    if (Check(TokenType::kNumber)) {
      double v = current_.number_value;
      Advance();
      values.push_back(std::make_unique<ConstantExpr>(v));
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
    throw std::runtime_error("Expected value");
  }

  void HandleRParen(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops) {
    while (!ops.empty() && ops.back().kind != OperatorEntry::Kind::kLParen) {
      ApplyOperator(values, ops.back());
      ops.pop_back();
    }
    if (ops.empty()) {
      throw std::runtime_error("Mismatched parentheses");
    }
    ops.pop_back();
    if (!ops.empty() && ops.back().kind == OperatorEntry::Kind::kFunc) {
      ApplyOperator(values, ops.back());
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
    throw std::runtime_error("Unexpected token");
  }

  void FinalizeExpression(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops) {
    while (!ops.empty()) {
      if (ops.back().kind == OperatorEntry::Kind::kLParen) {
        throw std::runtime_error("Mismatched parentheses");
      }
      ApplyOperator(values, ops.back());
      ops.pop_back();
    }
    if (values.size() != 1) {
      throw std::runtime_error("Invalid expression");
    }
  }

  ExprPtr ParseExpression() {
    std::vector<ExprPtr> values;
    std::vector<OperatorEntry> ops;
    values.reserve(16);
    ops.reserve(16);
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
    return std::move(values.back());
  }

  Lexer lexer_;
  Token current_;
  std::filesystem::path path_;
};

struct SharedExprFunctor {
  std::shared_ptr<Expr> ptr;
  double operator()(double x) const {
    return ptr->Eval(x);
  }
};

Function BuildFunction(const nlohmann::json &json, const std::filesystem::path &path) {
  const std::string expr = json.at("expression");
  Lexer lex(expr, path);
  Parser parser(std::move(lex), path);
  ExprPtr ast = parser.Parse();
  auto sp = std::shared_ptr<Expr>(std::move(ast));
  return Function{SharedExprFunctor{sp}};
}

double ComputeSafeStep(double requested_step, double span) {
  if (span <= 0.0) {
    return 1e-4;
  }
  if (requested_step > 0.0 && requested_step < span) {
    return requested_step;
  }
  double by_points = span / 1000.0;
  return std::max(by_points, 1e-4);
}

bool IsIncrementAcceptable(double value, double last_value, double dx, double value_threshold, double slope_threshold) {
  if (!std::isfinite(last_value)) {
    return true;
  }
  double diff = std::abs(value - last_value);
  double slope = (dx != 0.0) ? diff / std::abs(dx) : std::numeric_limits<double>::infinity();
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

bool ShouldAcceptRightEndpoint(double vr, double last_value, double value_threshold) {
  if (!std::isfinite(vr)) {
    return false;
  }
  if (!std::isfinite(last_value)) {
    return true;
  }
  double diff = std::abs(vr - last_value);
  if (diff > value_threshold) {
    return false;
  }
  if (diff > 1e5 * std::abs(last_value)) {
    return false;
  }
  return true;
}

ReferenceResult BruteForceReference(const Problem &p, double step) {
  double span = p.right - p.left;
  double safe_step = ComputeSafeStep(step, span);
  ReferenceResult r;
  r.argmin = p.left;
  r.value = std::numeric_limits<double>::infinity();
  double x = p.left;
  constexpr double kValueThreshold = 1e3;
  constexpr double kSlopeThreshold = 1e3;
  double last = std::numeric_limits<double>::quiet_NaN();
  double last_x = std::numeric_limits<double>::quiet_NaN();
  while (x <= p.right + 1e-12) {
    double v = p.func(x);
    bool ok = std::isfinite(v);
    if (ok && std::isfinite(last)) {
      double dx = x - last_x;
      if (!IsIncrementAcceptable(v, last, dx, kValueThreshold, kSlopeThreshold)) {
        ok = false;
      }
    }
    if (ok && v < r.value) {
      r.value = v;
      r.argmin = x;
    }
    if (std::isfinite(v)) {
      last = v;
      last_x = x;
    }
    x += safe_step;
  }
  double vr = p.func(p.right);
  if (ShouldAcceptRightEndpoint(vr, last, kValueThreshold) && vr < r.value) {
    r.value = vr;
    r.argmin = p.right;
  }
  return r;
}

std::vector<TestCase> LoadTestCasesFromData() {
  namespace fs = std::filesystem;
  fs::path json_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_global_search, "tests.json");
  std::ifstream file(json_path);
  if (!file) {
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
    t.name = item.at("name");
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
  std::string mpi_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchMPI>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchMPI::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);
  std::string seq_name =
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
  static std::string PrintTestParam(const TestType &t) {
    return t.name;
  }

 protected:
  void SetUp() override {
    test_case_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = test_case_.problem;
    reference_ = BruteForceReference(input_, test_case_.brute_force_step);
  }

  bool CheckTestOutputData(OutType &o) final {
    double dx = std::abs(o.argmin - reference_.argmin);
    double dv = std::abs(o.value - reference_.value);
    double dx_tol = test_case_.tolerance;
    double dv_tol = 5.0 * test_case_.tolerance;
    bool good = (dx <= dx_tol && dv <= dv_tol);
    bool cross = false;
    if (dv <= dv_tol) {
      double r1 = input_.func(o.argmin);
      double r2 = input_.func(reference_.argmin);
      cross = (std::abs(r1 - reference_.value) <= dv_tol) && (std::abs(r2 - o.value) <= dv_tol);
    }
    return good || cross;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  TestCase test_case_;
  InType input_;
  ReferenceResult reference_{};
};

namespace {

TEST_P(SizovDGlobalSearchFunctionalTests, FindsGlobalMinimum) {
  ExecuteTest(GetParam());
}

const auto kTests = LoadTestCasesFromData();
const auto kTasks = BuildTestTasks(kTests);
const auto kVals = ::testing::ValuesIn(kTasks);
const auto kName = SizovDGlobalSearchFunctionalTests::PrintFuncTestName<SizovDGlobalSearchFunctionalTests>;

INSTANTIATE_TEST_SUITE_P(GlobalSearch, SizovDGlobalSearchFunctionalTests, kVals, kName);

}  // namespace

}  // namespace sizov_d_global_search
